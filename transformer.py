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
	def __init__(self, head_length, window_sz, use_mask):		
		super(Atten_Head, self).__init__()

                '''        
                For comparing to Huang paper:
		       - head_length = Dh
		       - window_sz = L
                '''
                
		self.use_mask = use_mask
		
		self.K = self.add_weight("K", (head_length, head_length))
		self.V = self.add_weight("V", (head_length, head_length))
		self.Q = self.add_weight("Q", (head_length, head_length))
		
		# We need to initialize E matrix here for relative attention
		self.E = self.add_weight("E", (window_sz, head_length))
		
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
		V = tf.matmul(inputs_for_values, self.K)
		Q = tf.matmul(inputs_for_queries, self.K)

                # Applying relative embedding: skew(xQE^T)
                
                # rel_emb = tf.matmul(Q, tf.transpose(K, perm=[0,2,1]))
		# S_rel = skew(relative_emb)

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
                - Slice the matrix to the last L rows, and all columns

        :param rel_emb: tensor of [batch_size x window_sz, window_sz]
        :return: skewed tensor of [batch_size x window_sz, window_sz]
        """

        # Todo


class Multi_Headed(tf.keras.layers.Layer):
	def __init__(self, emb_sz, use_mask):
		super(Multi_Headed, self).__init__()

		self.h = 7 # Must be a factor of emb_sz
		self.size = emb_size
		self.use_mask = use_mask

		self.linear = tf.keras.layers.Dense(self.size)
		# Vectorize the head creation process
                # head_n = Atten_Head(self.size / self.h, self.size, self.use_mask)
                	

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

		# Split data

                # Run attention heads
                # Z_n = head_n(inputs_for_keys, inputs_for_values, inputs_for_queries)

                # Concatenate outputs

                # Apply linear layers
                # output = linear(concat)

                #return output
		return None


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
	def __init__(self, emb_sz, is_decoder):
		super(Transformer_Block, self).__init__()

		self.ff_layer = Feed_Forwards(emb_sz)
		self.self_atten = Multi_Headed(emb_sz,use_mask=is_decoder)
		self.is_decoder = is_decoder
		if self.is_decoder:
			self.self_context_atten = Multi_Headed(emb_sz,use_mask=False)

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

		with av.trans_block(self.is_decoder):
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
