import numpy as np
import tensorflow as tf
import transformer
import midi_to_encoding
from midi_to_fig import *
import sys
import random



class Transformer(tf.keras.Model):
    def __init__(self):

        super(Transformer, self).__init__()


        # Hyperparameters
        self.batch_size = 5
        self.embedding_size = 50
        self.hidden_layer_size = 50
        # 0.01 performs better than 0.001, try 0.005
        self.learning_rate = 0.01
        self.num_epochs = 1
		# max window size for full data set
        # self.window_size = 17729
		# temp window size for subset of data
        self.window_size = 1000
        self.space_index = 412
        self.padding_index = 413
        self.num_heads = 1
        self.vocab_size = 414


        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Define english and french embedding layers:
        self.encoder_emb = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.decoder_emb = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)

        # Positional embedding not necessary in this implementation (apparently)

        # Define encoder and decoder layers:
        self.encoder = transformer.Transformer_Block(self.embedding_size, False, self.window_size, num_heads=self.num_heads)
        self.decoder = transformer.Transformer_Block(self.embedding_size, True, self.window_size, num_heads=self.num_heads)

        # Define dense layer(s)
        self.dense_1 = tf.keras.layers.Dense(self.hidden_layer_size, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(self.vocab_size, activation="softmax")

    @tf.function
    def call(self, encoder_input, decoder_input):
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
        decoder_embedded = self.decoder_emb(decoder_input)

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


def train(model, train_data):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param train_data: (num_midis, window_size)
    :return: None
    """

    # Initializing masking function for later (may not be necessary)
    masking_func = np.vectorize(lambda x: x != model.padding_index)

    # Shuffling inputs
    order = tf.random.shuffle(range(len(train_data)))
    train_data = tf.gather(train_data, order)

    # Iterating over all inputs
    for i in range(0, len(order), model.batch_size):
        inputs = train_data[i : i + model.batch_size, :-1]
        labels = train_data[i : i + model.batch_size, 1:]

        # Ensuring full batch
        if(len(inputs) == model.batch_size):

            # Creating mask
            mask = masking_func(labels)

            # Forward pass
            with tf.GradientTape() as tape:
                probabilities = model(inputs, inputs)
                loss = model.loss_function(probabilities, labels, mask)
                print("Training loss is {}".format(loss))

            # Applying gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    pass

def test(model, test_data):
    """
    Runs through one epoch - all testing examples.

    :param model: the initialized model to use for forward and backward pass
    :param test_data:test data (all data for testing) of shape (num_midis, window_size)
    :returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set,
    e.g. (my_perplexity, my_accuracy)
    """

    # Initializing masking function for later
    masking_func = np.vectorize(lambda x: x != model.padding_index)

    # Initializing iterators
    symbol_count = 0
    plex_sum = 0
    accuracy_sum = 0

    for i in range(0, len(test_data), model.batch_size):
        inputs = test_data[i : i + model.batch_size, :-1]
        labels = test_data[i : i + model.batch_size, 1:]

        # Ensuring full batch
        if(len(labels) == model.batch_size):

            # Counting relevant metrics
            mask = masking_func(labels)
            for i in mask.flatten():
                if i: symbol_count += 1


            probabilities = model(inputs, inputs)
            plex_sum += model.loss_function(probabilities, labels, mask)
            accuracy_sum += model.accuracy_function(probabilities, labels, mask)

    # Calculating per symbol accuracy
    perplexity = np.exp(plex_sum / symbol_count)
    accuracy = accuracy_sum / int(len(test_data) / model.batch_size)

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

def main():

    print("Running preprocessing...")
    # Implement preprocessing
    train_data, test_data, val_data = midi_to_encoding.main()
    # turn the midi files into one-dimensional vectors, with space tokens in between each timestep
    train_data = format_data(train_data)
    test_data = format_data(test_data)
    val_data = format_data(val_data)

    model = Transformer()

    train_data = pad_data(17730, model.padding_index, train_data)
    test_data = pad_data(17730, model.padding_index, test_data)
    val_data = pad_data(17730, model.padding_index, val_data)

    print("Preprocessing complete.")
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)
	# slicing data to run on local, using the entire dataset causes memory issues
    train_data = train_data[:, :1000]
    test_data = test_data[:, :1000]
    # print((len(train_data), len(train_data[0])))
    train_data = np.hstack((train_data, np.full((train_data.shape[0], 1), model.space_index)))
    test_data = np.hstack((test_data, np.full((test_data.shape[0], 1), model.space_index)))

    print(train_data.shape)

    # Train and Test Model
    for i in range(model.num_epochs):
        train(model, train_data)
        plex, acc = test(model, test_data)

        # Printing resulatant perplexity and accuracy
        print("Epoch:", i)
        print("Perplexity", plex, "Accuracy", acc)

    # Run model to create mididata


if __name__ == '__main__':
    main()

