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
    def __init__(self, vocab_size):

        super(Transformer, self).__init__()


        # Hyperparameters
        self.batch_size = tf.Variable(30, trainable=False) #2
        self.embedding_size = tf.Variable(50, trainable=False)
        self.hidden_layer_size = tf.Variable(200, trainable=False)
        # 0.01 performs better than 0.001, try 0.005
        self.learning_rate = tf.Variable(0.01, trainable=False)
        self.num_epochs = tf.Variable(1, trainable=False)
        # max window size for full data set
        
        # temp window size for subset of data
        self.window_size = tf.Variable(400, trainable=False) #3999
        self.padding_index = tf.Variable(vocab_size - 1, trainable=False)
        self.num_heads = tf.Variable(2, trainable=False)
        self.vocab_size = tf.Variable(vocab_size, trainable=False)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Define english and french embedding layers:
        self.encoder_emb = tf.keras.layers.Embedding(self.vocab_size.numpy(), self.embedding_size.numpy())
        self.decoder_emb = tf.keras.layers.Embedding(self.vocab_size.numpy(), self.embedding_size.numpy())

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

        return tf.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs, False), mask))

def train(note_model, vel_model, time_model):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param train_data: (num_midis, window_size)
    :return: None
    """
    
    folder_list = os.listdir("dataset-v4/train")
    random.shuffle(folder_list)
    length = len(os.listdir("dataset-v4/train"))
    
    #loss_array = np.asarray([])
    steps = 0
    
    for i in range(0, length - note_model.batch_size.numpy(), note_model.batch_size.numpy()):
        steps += 1
        print("Step: {}".format(steps))
        note_data, vel_data, time_data = process(note_model, vel_model, time_model, i, "dataset-v4/train", folder_list)

        train_loop(note_model, note_data)
        train_loop(vel_model, vel_data)
        train_loop(time_model, time_data)
        
        note_model.save_weights("note_weights", overwrite=True)
        vel_model.save_weights("vel_weights", overwrite=True)
        time_model.save_weights("time_weights", overwrite=True)
        
        if steps >= 2200:
            break
    #return loss_array
    pass
        

def train_loop(model, train_data):
    masking_func = np.vectorize(lambda x: x != model.padding_index.numpy())
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
            #loss_array = np.append(loss_array, loss)
            print("Training loss is {}".format(loss))
            
        # Applying gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        #np.save("loss_array_test", loss_array)
    pass
        

def process(note_model, vel_model, time_model, j, folder, folder_list):
    
    data = get_data_split(folder, folder_list, j, note_model.batch_size.numpy())

    # separating note on/offs, velocities, and time steps
    notes = []
    vels = []
    time = []
    for example in data:
        notes.append(example[:,:256])
        vels.append(example[:,256:288])
        time.append(example[:,288:])
    
    # note formatting
    notes = format_data(notes)
    notes = pad_data(7000, note_model.padding_index.numpy(), notes)

    # velocity formatting
    vels = format_data(vels)
    vels = pad_data(7000, vel_model.padding_index.numpy(), vels)

    # time formatting
    time = format_data(time)
    time = pad_data(7000, time_model.padding_index.numpy(), time)
    
    # slicing data to run on local, using the entire dataset causes memory issues
    notes = notes[:, :note_model.window_size.numpy() + 1]
    vels = vels[:, :vel_model.window_size.numpy() + 1]
    time = time[:, :time_model.window_size.numpy() + 1]
    
    return notes, vels, time

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
    return np.asarray(array)

'''
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
'''

def intersect(carrier, modulator):
    values = [val for val in carrier if val in modulator]
    for i in range(len(values)):
        values[i] -= min(modulator)
    return values

def model_run(model, seq, i):
    model_input = np.asarray([seq])
    probs = model(model_input)[0][i]
    return probs.numpy()

def generate_sequence(note_model, vel_model, time_model, start_sequence, length):
    window_size = note_model.window_size.numpy()

    # separating note, velocity, time information from the start sequence
    note_seq = intersect(start_sequence, list(range(256)))
    note_i = len(note_seq)
    note_seq = list(pad_data(window_size, note_model.padding_index.numpy(), [note_seq])[0])
    
    vel_seq = intersect(start_sequence, list(range(256, 288)))
    vel_i = len(vel_seq)
    vel_seq = list(pad_data(window_size, vel_model.padding_index.numpy(), [vel_seq])[0])

    time_seq = intersect(start_sequence, list(range(288, 413)))
    time_i = len(time_seq)
    time_seq = list(pad_data(window_size, time_model.padding_index.numpy(), [time_seq])[0])
    
    result_vectors = [start_sequence]
    k = 10
    p = 0.92
    pause_prob = 0.8
    on_prob = 0.4
    
    # loop until sequence is of the given length
    while len(result_vectors) < length:

        curr_event = []
        
        # call model on the sequence to get the probability of the next 'word'
        note_probs = model_run(note_model, note_seq, note_i)
        vel_probs = model_run(vel_model, vel_seq, vel_i)
        
        note_on = note_probs[:128]
        note_off = note_probs[128:]
        
        if np.random.choice([0, 1], p=[pause_prob, 1 - pause_prob]):
            if np.random.choice([1,0], p=[on_prob, 1 - on_prob]):
                note_event = selection(k, p, note_on, 0)
                vel_event = selection(k, p, vel_probs, 256)
                curr_event.extend([note_event, vel_event])
            else:
                note_event = selection(k, p, note_off, 128)
                vel_event = 256
                curr_event.extend([note_event, vel_event])

            if note_i < window_size - 1:
                note_seq[note_i] = note_event
                note_i += 1
            else:
                note_seq.append(note_event)
                note_seq = note_seq[-window_size:]

            if vel_i < window_size - 1:
                vel_seq[vel_i] = vel_event - 256
                vel_i += 1
            else:
                vel_seq.append(vel_event - 256)
                vel_seq = vel_seq[-window_size:]



        time_probs = model_run(time_model, time_seq, time_i)
        time_event = selection(k, p, time_probs, 288)
        curr_event.append(time_event)
        
        if time_i < window_size - 1:
            time_seq[time_i] = time_event - 288
            time_i += 1
        else:
            time_seq.append(time_event - 288)
            time_seq = time_seq[-window_size:]
        
        #if note_i >= window_size - 1: result_vectors.append(curr_event)
        result_vectors.append(curr_event)
        
    return result_vectors
    

def selection(k, p, note_prob, shift):
    
    note_argsort = note_prob.argsort()[::-1]
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
    
        vectors.append(vector)

    return np.asarray(vectors)

def singleCall(note_model, vel_model, time_model):
    folder_list = os.listdir("dataset-v4/train")
    random.shuffle(folder_list)
    note_data, vel_data, time_data = process(note_model, vel_model, time_model, 0, "dataset-v4/train", folder_list)

    train_loop(note_model, note_data)
    train_loop(vel_model, vel_data)
    train_loop(time_model, time_data)
    pass
    

def main():
    note_model = Transformer(256)
    vel_model = Transformer(32)
    time_model = Transformer(125)
    singleCall(note_model, vel_model, time_model)
    
    #model.load_weights("/home/dante_rousseve/model_midtrain_weights_1")
    note_model.load_weights("note_weights")
    vel_model.load_weights("vel_weights")
    time_model.load_weights("time_weights")
    
    # Train and Test Model
    if False:
        for i in range(note_model.num_epochs.numpy()):
            loss_array = train(note_model, vel_model, time_model)
            #plex, acc = test(model)

            # Printing resulatant perplexity and accuracy
            #print("Epoch:", i)
            #print("Perplexity", plex, "Accuracy", acc)

            '''
            print("Saving Loss Graph")
            plt.plot(loss_array)
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.savefig("Loss v. Training Step in Epoch {}.png".format(i))
            '''

    #model.save_weights("model_weights", overwrite=True)
    #model.save("saved_model")
    for i in range(1):
        print("Generating midi #{}".format(i+1))
        # Run model to create mididata
        start_seq = [50, 53, 57, 62, 60, 58, 60, 62, 50, 62, 280, 400, 401, 402, 403, 404]
    
        #start_seq = start_seqs[i % 5]

        sequence = np.asarray(generate_sequence(note_model, vel_model, time_model, start_seq, 1000))
        vectors = convert_to_vectors(sequence)
        print("Encoding Midi")
        midi = encoding_to_midi(vectors, "midi_out_{}.midi".format(i+1))

if __name__ == '__main__':
    main()



