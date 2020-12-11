import numpy as np
import tensorflow as tf
import transformer
from midi_to_encoding import *
from midi_to_fig import *
import sys
import random
import os
from utils import encoding_to_midi
import matplotlib.pyplot as plt



class Transformer(tf.keras.Model):
    def __init__(self):

        super(Transformer, self).__init__()


        # Hyperparameters
        self.batch_size = tf.Variable(2, trainable=False)
        self.embedding_size = tf.Variable(50, trainable=False)
        self.hidden_layer_size = tf.Variable(50, trainable=False)
        # 0.01 performs better than 0.001, try 0.005
        self.learning_rate = tf.Variable(0.01, trainable=False)
        self.num_epochs = tf.Variable(1, trainable=False)
        # max window size for full data set
        
        # temp window size for subset of data
        self.window_size = tf.Variable(6999, trainable=False)
        self.space_index = tf.Variable(412, trainable=False)
        self.padding_index = tf.Variable(413, trainable=False)
        self.num_heads = tf.Variable(2, trainable=False)
        self.vocab_size = tf.Variable(414, trainable=False)

        # generation hyper params
        self.max_phrase_length = tf.Variable(7, trainable=False)
        self.phrase_rand_amount = tf.Variable(3, trainable=False)

        self.train = tf.Variable(False, trainable=False)



        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Define english and french embedding layers:
        self.encoder_emb = tf.keras.layers.Embedding(self.vocab_size.numpy(), self.embedding_size.numpy())
        self.decoder_emb = tf.keras.layers.Embedding(self.vocab_size.numpy(), self.embedding_size.numpy())

        # Positional embedding not necessary in this implementation (apparently)

        # Define encoder and decoder layers:
        self.encoder = transformer.Transformer_Block(self.embedding_size.numpy(), False, self.window_size.numpy(), num_heads=self.num_heads.numpy())
        self.decoder = transformer.Transformer_Block(self.embedding_size.numpy(), True, self.window_size.numpy(), num_heads=self.num_heads.numpy())

        # Define dense layer(s)
        self.dense_1 = tf.keras.layers.Dense(self.hidden_layer_size.numpy(), activation="relu")
        self.dense_2 = tf.keras.layers.Dense(self.vocab_size.numpy(), activation="softmax")

    @tf.function
    def call(self, encoder_input):
        """
        :param encoder_input:
        :param decoder_input:
        :return prbs: Probabilities as a tensor, [batch_size x window_size x input_size]
        """

        #1) Embed the encoder_input

        encoder_embedded = self.encoder_emb(encoder_input)

        #2) Pass the encoder_input embeddings to the encoder
        context = self.encoder(encoder_embedded)

        #3) Embed the decoder_input
        decoder_embedded = self.decoder_emb(encoder_input)

        #4) Pass the decoder_input embeddings and result of the encoder to the decoder
        decoded = self.decoder(decoder_embedded, context)

        #5) Apply dense layers to the decoder out to generate probabilities
        result = self.dense_1(decoded)
        result = self.dense_2(result)

        return result

    def accuracy_function(self, prbs, labels, mask):
        """
        Computes the batch accuracy

        :param prbs:  float tensor, prediction probabilities [batch_size x window_size x input_size]
        :param labels:  integer tensor, prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """

                # Masking may not be necessary
                
        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
        return accuracy


    def loss_function(self, prbs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x input_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        # Implement negative log likelyhood
        # Masking may not be necessary

        return tf.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs, False), mask))

    # @av.call_func
    # def __call__(self, *args, **kwargs):
    # 	return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)


def train(model):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param train_data: (num_midis, window_size)
    :return: None
    """

    
    # Initializing masking function for later (may not be necessary)
    masking_func = np.vectorize(lambda x: x != model.padding_index.numpy())
    length = len(os.listdir("dataset-v4/train"))
    global_max = 0
    loss_array = np.asarray([])
    steps = 0
    # Iterating over all inputs
    for i in range(0, length - model.batch_size.numpy(), model.batch_size.numpy()):
        steps += 1
        print("Step: {}".format(steps))
        train_data, batch_max = process(model, i, "dataset-v4/train")
        global_max = max(global_max, batch_max)
        
        inputs = train_data[:, :-1]
        labels = train_data[:, 1:]

        # Ensuring full batch
        if(len(inputs) == model.batch_size.numpy()):

            # Creating mask
            mask = masking_func(labels)

            # Forward pass
            with tf.GradientTape() as tape:
                probabilities = model(inputs)
                loss = model.loss_function(probabilities, labels, mask)
                loss_array = np.append(loss_array, loss)
                print("Training loss is {}".format(loss))
                
            # Applying gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            model.save_weights("model_midtrain_weights_test", overwrite=True)
            np.save("loss_array_test", loss_array)

        if steps >= 2200:
            break
    return loss_array

def process(model, j, folder):
    folder_list = os.listdir(folder)
    folder_list = random.shuffle(folder_list)
    data = get_data_split(folder, folder_list, j, model.batch_size.numpy())
    # turn the midi files into one-dimensional vectors, with space tokens in between each timestep
    data = format_data(data)

    maximum = np.max(list(map(lambda x: len(x), data)))
    print(maximum)
    data = pad_data(7000, model.padding_index.numpy(), data)
    data = np.asarray(data)
    #data = data.reshape((-1, model.window_size.numpy() + 1))
    #print(data.shape)
    # slicing data to run on local, using the entire dataset causes memory issues
    #data = data[:, :model.window_size.numpy()]
    
   # data = np.hstack((data, np.full((data.shape[0], 1), model.space_index.numpy())))
    
    return data, maximum


def test(model):
    """
    Runs through one epoch - all testing examples.

    :param model: the initialized model to use for forward and backward pass
    :param test_data:test data (all data for testing) of shape (num_midis, window_size)
    :returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set,
    e.g. (my_perplexity, my_accuracy)
    """

    # Initializing masking function for later
    masking_func = np.vectorize(lambda x: x != model.padding_index.numpy())

    # Initializing iterators
    symbol_count = 0
    plex_sum = 0
    accuracy_sum = 0
    global_max = 0

    length = len(os.listdir("dataset-v4/test"))

    for i in range(0, length - model.batch_size.numpy(), model.batch_size.numpy()):
        test_data, batch_max = process(model, i, "dataset-v4/test")
        global_max = max(global_max, batch_max)
        
        inputs = test_data[:, :-1]
        labels = test_data[:, 1:]

        # Ensuring full batch
        if(len(labels) == model.batch_size.numpy()):

            # Counting relevant metrics
            mask = masking_func(labels)
            for i in mask.flatten():
                if i: symbol_count += 1


            probabilities = model(inputs)
            plex_sum += model.loss_function(probabilities, labels, mask)
            accuracy_sum += model.accuracy_function(probabilities, labels, mask)

    # Calculating per symbol accuracy
    perplexity = np.exp(plex_sum / symbol_count)
    accuracy = accuracy_sum / int(len(test_data) / model.batch_size.numpy())

    return (perplexity, accuracy)

def format_data(array):
    for i in range(len(array)):
        midi = array[i]
        index_list = []
        for j in range(len(midi)):
            indices = np.where(midi[j] == 1.0)
            index_list = np.append(index_list, indices)

        array[i] = index_list

    return array

def pad_data(window_size, padding_index, array):
    for i in range(len(array)):
        midi = array[i]
        if len(midi) < window_size:
            missing_steps = window_size - len(midi)
            padding = np.full((missing_steps), padding_index)
            midi = np.append(midi, padding)
            array[i] = midi
    return array

def generate_sequence(model, start_sequence, length):
    padded_sequence = np.asarray(pad_data(model.window_size.numpy(), model.padding_index.numpy(), [start_sequence]))[0]
    # print(padded_sequence.shape)
    final_sequence = start_sequence
    result_vectors = [start_sequence]
    k = 30
    p = 0.80
    seq_index = len(start_sequence)
    
    # loop until sequence is of the given length
    while len(result_vectors) < length:

        curr_event = []
        
        # call model on the sequence to get the probability of the next 'word'
        model_input = np.asarray([padded_sequence])
        probs = model(model_input)[0][seq_index]
        index_array = probs.numpy()
        
        note_on = index_array[:128]
        note_off = index_array[128:256]
        velocity = index_array[256:288]
        time_step = index_array[288:413]
        
        #print(np.sum(note_on), np.sum(note_off), np.sum(velocity), np.sum(time_step))
        #print(probs)
        if np.sum(note_on) > np.sum(note_off):
            note_event = selection(k, p, note_on, 0)
            vel = selection(32, p, velocity, 256)
            
            final_sequence.extend([note_event, vel])
            curr_event.extend([note_event, vel])

        else: 
            note_event = selection(k, p, note_off, 128)

            final_sequence.append(note_event)
            curr_event.append(note_event)

        time = selection(k, p, time_step, 288)

        final_sequence.append(time)
        curr_event.append(time)
        
        print(curr_event)
        if seq_index >= model.window_size.numpy() - 5:
            result_vectors.append(curr_event)

        size = len(curr_event)
        
        if seq_index < model.window_size.numpy() - size:
            padded_sequence[seq_index:seq_index + size] = curr_event
            seq_index += size
        else:
            padded_sequence = np.asarray(final_sequence[-model.window_size.numpy():])
            
    return result_vectors
    

def selection(k, p, note_prob, shift):
    
    note_argsort = note_prob.argsort()[::-1]
    if not shift:
        print(note_argsort)
    note_indeces = note_argsort[:k] + shift
    note_prob = tf.gather(tf.constant(note_prob), note_indeces - shift)
    note_prob = tf.nn.softmax(note_prob)

    prob_sum = 0
    indices = []
    i = 0
    while prob_sum < p:
        index = note_indeces[i]
        prob = note_prob[i]
        prob_sum += prob
        indices.append(index)
        i += 1


    next_event = random.choices(indices, note_prob[:i], k=1)[0]
    return next_event

def convert_to_vectors(sequence, vector_length=413):
    #return np.asarray(list(map(lambda x: tf.one_hot(x, vector_length), sequence)))
    vectors = []
    for i in range(len(sequence)):
        vector = np.zeros(vector_length)
        vector[sequence[i]] = 1
    
        print(vector)
        vectors.append(vector)

    return np.asarray(vectors)

def singleCall(model):
    masking_func = np.vectorize(lambda x: x != model.padding_index.numpy())
    train_data, batch_max = process(model, 0, "dataset-v4/train")
    inputs = train_data[:, :-1]
    labels = train_data[:, 1:]
    mask = masking_func(labels)

    # Forward pass
    with tf.GradientTape() as tape:
        probabilities = model(inputs)
        loss = model.loss_function(probabilities, labels, mask)

    # Applying gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    pass
    

def main():
    model = Transformer()
    singleCall(model)
    
    model.load_weights("/home/dante_rousseve/model_midtrain_weights_1")
    
    # Train and Test Model
    if False:
        for i in range(model.num_epochs.numpy()):
            loss_array = train(model)
            #plex, acc = test(model)

            # Printing resulatant perplexity and accuracy
            #print("Epoch:", i)
            #print("Perplexity", plex, "Accuracy", acc)

            print("Saving Loss Graph")
            plt.plot(loss_array)
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.savefig("Loss v. Training Step in Epoch {}.png".format(i))

    #model.save_weights("model_weights", overwrite=True)
    #model.save("saved_model")
    for i in range(15):
        print("Generating midi #{}".format(i+1))
        # Run model to create mididata
        start_seq = [70, 280, 346]
    
        #start_seq = start_seqs[i % 5]

        sequence = np.asarray(generate_sequence(model, start_seq, 1500))
        vectors = convert_to_vectors(sequence)
        print(vectors.shape)
        print("Encoding Midi")
        midi = encoding_to_midi(vectors, "midi_out_{}.midi".format(i+1))

if __name__ == '__main__':
    main()



