import tensorflow as tf
import numpy as np


def Attention_Matrix(K, Q, S_rel, use_mask=False):
    """
    Returns the attention matrix for a single attention head

    :param K: is [batch_size x window_sz x head_length]
    :param Q: is [batch_size x window_sz x head_length]
    :param S_rel: is [batch_size x window_sz x window_sz]
    :return: attention matrix
    """


    window_size_queries = Q.get_shape()[1] # window size of queries
    window_size_keys = K.get_shape()[1] # window size of keys
    mask = tf.convert_to_tensor(value=np.transpose(np.tril(np.ones((window_size_queries,window_size_keys))*np.NINF,-1),(1,0)),dtype=tf.float32)
    atten_mask = tf.tile(tf.reshape(mask,[-1,window_size_queries,window_size_keys]),[tf.shape(input=K)[0],1,1]) # creates upper triangular mask for decoder self-attention

    # Computing attention weights using queries and key matrices (if use_mask==True, adding the attention mask before softmax)
    K_T = tf.transpose(K, perm=[0,2,1])
    weights = (tf.matmul(Q, K_T) + S_rel) / np.sqrt(window_size_keys)
    if(use_mask): weights += atten_mask
    weights = tf.nn.softmax(weights)

    return weights


class Atten_Head(tf.keras.layers.Layer):
    """
    Had to add parameters here [Replaced 'head_length' with params for 'input_size' and 'output_size']
    'head_length' is 'input_size' and is variable depending on whether the 'emb_sz' evenly divides into 'num_heads'
    This may be the wrong way to go about it, but this retains the most similarity to the hw4 code.
    """
    def __init__(self, input_size, output_size, window_sz, use_mask):
        super(Atten_Head, self).__init__()
        self.use_mask = use_mask

        self.K = self.add_weight("K", (input_size, output_size))
        self.V = self.add_weight("V", (input_size, output_size))
        self.Q = self.add_weight("Q", (input_size, output_size))

        # We need to initialize E matrix here for relative attention
        self.E = self.add_weight("E", (window_sz, output_size))

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

        """
        This function runs a single attention head.

        :param inputs_for_keys: tensor of [batch_size x window_sz x head_length ]
        :param inputs_for_values: tensor of [batch_size x window_sz x head_length ]
        :param inputs_for_queries: tensor of [batch_size x window_sz x head_length ]
        :return: tensor of [BATCH_SIZE x window_sz x head_length ]
        """

        # Creating Key, Value, and Query matrices
        K = tf.matmul(inputs_for_keys, self.K)
        V = tf.matmul(inputs_for_values, self.V)
        Q = tf.matmul(inputs_for_queries, self.Q)
        print("Q shape is {}, E shape is {}".format(Q.shape, self.E.shape))
        rel_emb = tf.matmul(Q, tf.transpose(self.E, perm=[1,0]))
        S_rel = skew(rel_emb)

        atten_weights = Attention_Matrix(K, Q, S_rel, self.use_mask)
        Z = tf.matmul(atten_weights, V)
        return Z


def skew(rel_emb):
    """
    This function skews the relative embeddings to make them positionally appropriate.
    Follows the global skewing procedure outlined in section 3.4.1 of the Huang paper

    Procedure:
            - Pad a column to the leftmost position
            - Reshape the matrix to shape (L + 1, L)
            - Slice the matrix to the last L rows

    :param rel_emb: tensor of [batch_size x window_sz, window_sz]
    :return: skewed tensor of [batch_size x window_sz, window_sz]
    """
    # rel_emb = np.asarray(rel_emb)
    batch_size = len(rel_emb)
    window_size = len(rel_emb[0])
    input_size = len(rel_emb[0][0])
    padded = tf.pad(rel_emb, ((0,0),(0,0),(1,0)))
    reshaped = tf.reshape(padded, (batch_size, input_size + 1, window_size))
    return reshaped[:,1:,:]


class Multi_Headed(tf.keras.layers.Layer):
    def __init__(self, emb_sz, use_mask, window_size, num_heads):
        super(Multi_Headed, self).__init__()

        self.h = num_heads # Does not have to be an exact factor of emb_sz
        self.size = emb_sz
        self.use_mask = use_mask
        self.window_size = window_size
        # Initialize heads
        self.attention_heads = []

        if num_heads == 1:
            head = Atten_Head(emb_sz, emb_sz, window_size, use_mask)
            self.attention_heads.append(head)

        else:
            # Split emb_sz into 'num_heads' groups, where the last head holds the elements that don't evenly divide
            output_size = emb_sz // (num_heads - 1)
            extra = emb_sz % (num_heads - 1)
            # create heads
            for i in range(num_heads):
                # if we're at the last head...
                if i == num_heads - 1:
                    # and there is uneven division
                    if extra is not 0:
                        # create a head with the extra elements
                        head = Atten_Head(emb_sz, extra, window_size, use_mask)
                # otherwise create a evenly divided head
                else:
                    head = Atten_Head(emb_sz, output_size, window_size, use_mask)
                # add head to the head list
                self.attention_heads.append(head)


    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        This functions runs a multiheaded attention layer.

        Procedure:
            - Split data for h different heads of size embed_sz/h
            - Apply different attention heads
            - Concatenate the outputs of these heads together
            - Apply a linear layer

        :param inputs_for_keys: tensor of [batch_size x window_sz x input_size ]
        :param inputs_for_values: tensor of [batch_size x window_sz x input_size ]
        :param inputs_for_queries: tensor of [batch_size x window_sz x input_size ]
        :return: tensor of [BATCH_SIZE x window_sz x output_size ]
        """

        attentions = []
        # call each of the attention heads with the inputs
        for i in range(len(self.attention_heads)):
            head = self.attention_heads[i]
            # compute individual attentions
            attention = head.call(inputs_for_keys, inputs_for_values, inputs_for_queries)
            # append to attention list
            attentions.append(attention)

        # concatentate along axis 2, to recombine into the full attention tensor
        attentions = tf.concat(attentions, axis=2)
        return attentions


class Feed_Forwards(tf.keras.layers.Layer):
    def __init__(self, emb_sz):
        super(Feed_Forwards, self).__init__()

        self.layer_1 = tf.keras.layers.Dense(emb_sz,activation='relu')
        self.layer_2 = tf.keras.layers.Dense(emb_sz)

    @tf.function
    def call(self, inputs):
        """
        This functions creates a feed forward network as described in 3.3
        https://arxiv.org/pdf/1706.03762.pdf

        Requirements:
        - Two linear layers with relu between them

        :param inputs: input tensor [batch_size x window_size x embedding_size]
        :return: tensor [batch_size x window_size x embedding_size]
        """
        layer_1_out = self.layer_1(inputs)
        layer_2_out = self.layer_2(layer_1_out)
        return layer_2_out

class Transformer_Block(tf.keras.layers.Layer):
    def __init__(self, emb_sz, is_decoder, window_size, num_heads):
        super(Transformer_Block, self).__init__()

        self.ff_layer = Feed_Forwards(emb_sz)
        self.self_atten = Multi_Headed(emb_sz, use_mask=is_decoder, window_size=window_size, num_heads=num_heads)
        self.is_decoder = is_decoder
        if self.is_decoder:
            self.self_context_atten = Multi_Headed(emb_sz,use_mask=False, window_size=window_size, num_heads=num_heads)

        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

    @tf.function
    def call(self, inputs, context=None):
        """
        This functions calls a transformer block.

        There are two possibilities for when this function is called.
            - if self.is_decoder == False, then:
            1) compute unmasked attention on the inputs
            2) residual connection and layer normalization
            3) feed forward layer
            4) residual connection and layer normalization

            - if self.is_decoder == True, then:
            1) compute MASKED attention on the inputs
            2) residual connection and layer normalization
            3) computed UNMASKED attention using context
            4) residual connection and layer normalization
            5) feed forward layer
            6) residual layer and layer normalization

        :param inputs: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ]
        :context: tensor of [BATCH_SIZE x FRENCH_WINDOW_SIZE x EMBEDDING_SIZE ] or None
            default=None, This is context from the encoder to be used as Keys and Values in self-attention function
        """

        atten_out = self.self_atten(inputs,inputs,inputs)
        atten_out+=inputs
        atten_normalized = self.layer_norm(atten_out)

        if self.is_decoder:
            assert context is not None,"Decoder blocks require context"
            context_atten_out = self.self_context_atten(context,context,atten_normalized)
            context_atten_out+=atten_normalized
            atten_normalized = self.layer_norm(context_atten_out)

        ff_out=self.ff_layer(atten_normalized)
        ff_out+=atten_normalized
        ff_norm = self.layer_norm(ff_out)

        return tf.nn.relu(ff_norm)
