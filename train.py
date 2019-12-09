from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import os
import sys
import time
import tensorflow as tf
import tempfile


from utils import load_text, split_input_target, loss
from model import GRU_model, LSTM_model

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("dataset", "Data/republic_clean.txt", "Data file for the task")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim",512,"Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("RNN_units",1024,"Dimensionality of character embedding (default: 128)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs ")
tf.flags.DEFINE_integer("buffer_size", 1000, "Buffer size for shuffling ")


FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

def preprocess():
    """
    
    """
    data = FLAGS.dataset
    text = load_text(data)
    vocab = sorted(set(text))

    # Creating a mapping from unique characters to indices
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    # The maximum length sentence we want for a single input in characters
    seq_length = 150
    examples_per_epoch = len(text)//(seq_length+1)

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    # The batch method lets us easily convert these individual characters to sequences of the desired size.
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    # For each sequence, duplicate and shift it to form the input and target text by using the map method to apply a simple function to each batch:
    dataset = sequences.map(split_input_target)

    # shuffle the data and pack it into batches.
    dataset = dataset.shuffle(FLAGS.buffer_size).batch(FLAGS.batch_size, drop_remainder=True)
    
    return vocab, dataset

def train(dataset, model, vocab):
    # Training
    # ==================================================
    
    model = model(
        vocab_size = len(vocab),
        embedding_dim=FLAGS.embedding_dim,
        rnn_units=FLAGS.RNN_units,
        batch_size=FLAGS.batch_size)

    logdir = tempfile.mkdtemp()
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)]
    model.compile(optimizer='adam', loss=loss)

    history = model.fit(dataset, epochs = FLAGS.num_epochs, callbacks= callbacks)
    
    # Save the original model for size comparison later

    _ , keras_file = tempfile.mkstemp('.h5')
    print('Saving model to: ', keras_file)
    tf.keras.models.save_model(model, keras_file, include_optimizer=False)
    return model 

    
def main(argv=None):
    vocab, dataset  = preprocess()
    train(dataset, GRU_model, vocab)
if __name__ == '__main__':
    tf.app.run()
