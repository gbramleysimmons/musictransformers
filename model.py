import numpy as np
import tensorflow as tf
import transformer
from midi_to_encoding import *
from midi_to_fig import *
import sys
import random



class Transformer(tf.keras.Model):
	def __init__(self, input_size):

		super(Transformer, self).__init__()

		self.input_size = input_size

		# Hyperparameters
		self.batch_size = 50
		self.embedding_size = 50
		self.hidden_layer_size = 50
		self.learning_rate = 0.001
		self.num_epochs = 1
		self.window_size = 50
		

		# Optimizer
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

		# Define english and french embedding layers:
		self.encoder_emb = tf.keras.layers.Embedding(self.input_size, self.embedding_size)
		self.decoder_emb = tf.keras.layers.Embedding(self.input_size, self.embedding_size)
		
		# Positional embedding not necessary in this implementation (apparently)

		# Define encoder and decoder layers:
		self.encoder = transformer.Transformer_Block(self.embedding_size, False)
		self.decoder = transformer.Transformer_Block(self.embedding_size, True)
		
		# Define dense layer(s)
		self.dense_1 = tf.keras.layers.Dense(self.hidden_layer_size, activation="relu")
		self.dense_2 = tf.keras.layers.Dense(self.input_size, activation="softmax")

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

	@av.call_func
	def __call__(self, *args, **kwargs):
		return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)


def train(model, train_, train_, eng_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_french: french train data (all data for training) of shape (num_sentences, 14)
	:param train_english: english train data (all data for training) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: None
	"""





        """
        This function will be overhauled once we have a better understanding of preprocessing outputs

        Right now it is till in terms of Assignment 4
        """



	

	# Initializing masking function for later (may not be necessary)
	masking_func = np.vectorize(lambda x: x != eng_padding_index)
	
	# Shuffling inputs
	order = tf.random.shuffle(range(len(train_english)))
	train_french = tf.gather(train_french, order)
	train_english = tf.gather(train_english, order)

	# Iterating over all inputs (given 1 epoch)
	for i in range(0, len(order), model.batch_size):
		english_decoder = train_english[i : i + model.batch_size, :-1]
		english_loss = train_english[i : i + model.batch_size, 1:]
		french_batch = train_french[i : i + model.batch_size]

		# Ensuring full batch
		if(len(english_loss) == model.batch_size):

			# Creating mask
			mask = masking_func(english_loss)

			# Forward pass
			with tf.GradientTape() as tape:
				probabilities = model(french_batch, english_decoder)
				loss = model.loss_function(probabilities, english_loss, mask)

			# Applying gradients
			gradients = tape.gradient(loss, model.trainable_variables)
			model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	pass

def test(model, test_french, test_english, eng_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_french: french test data (all data for testing) of shape (num_sentences, 14)
	:param test_english: english test data (all data for testing) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
	e.g. (my_perplexity, my_accuracy)
	"""

	# Initializing masking function for later
	masking_func = np.vectorize(lambda x: x != eng_padding_index)

	# Initializing iterators
	symbol_count = 0
	plex_sum = 0
	accuracy_sum = 0

	for i in range(0, len(test_english), model.batch_size):
		english_decoder = test_english[i : i + model.batch_size, :-1]
		english_loss = test_english[i : i + model.batch_size, 1:]
		french_batch = test_french[i : i + model.batch_size]
		# Ensuring full batch
		if(len(english_loss) == model.batch_size):

			# Counting relevant metrics
			mask = masking_func(english_loss)
			for i in mask.flatten():
				if i: symbol_count += 1
				
				
			probabilities = model(french_batch, english_decoder)
			plex_sum += model.loss_function(probabilities, english_loss, mask)
			accuracy_sum += model.accuracy_function(probabilities, english_loss, mask)

	# Calculating per symbol accuracy
	perplexity = np.exp(plex_sum / symbol_count)
	accuracy = accuracy_sum / int(len(test_english) / model.batch_size)

	return (perplexity, accuracy)

def main():
        
	print("Running preprocessing...")
	# Implement preprocessing

	# Set input_size to the length of inputs (should be 413 iirc)
	input_size = 413
	print("Preprocessing complete.")

	model = Transformer(input_size)
		
	# Train and Test Model
	for i in range(model.num_epochs):
                train(model, train_french, train_english, eng_padding_index)
                plex, acc = test(model, test_french, test_english, eng_padding_index)

                # Printing resulatant perplexity and accuracy
                print("Epoch:", i)
                print("Perplexity", plex, "Accuracy", acc)
	
	# Run model to create mididata


if __name__ == '__main__':
	main()

