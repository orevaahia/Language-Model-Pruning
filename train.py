from __future__ import absolute_import, division, print_function, unicode_literals

import tensorboard
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
tf.app.flags.DEFINE_string("dataset", "Data/republic_clean.txt", "Data file for the task")

# Model Hyperparameters
tf.app.flags.DEFINE_integer("embedding_dim",512,"Dimensionality of character embedding (default: 128)")
tf.app.flags.DEFINE_integer("RNN_units",1024,"Dimensionality of character embedding (default: 128)")

# Training parameters
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs ")
tf.app.flags.DEFINE_integer("buffer_size", 1000, "Buffer size for shuffling ")
tf.app.flags.DEFINE_boolean("drop_remainder", True, "Drop last batch if it has fewer batch size elements")


FLAGS = tf.app.flags.FLAGS
FLAGS(sys.argv)

def preprocess():
    """
    Prepare data 
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
    sequences = char_dataset.batch(seq_length+1, drop_remainder=FLAGS.drop_remainder)

    # For each sequence, duplicate and shift it to form the input and target text by using the map method to apply a simple function to each batch:
    dataset = sequences.map(split_input_target)

    # shuffle the data and pack it into batches.
    dataset = dataset.shuffle(FLAGS.buffer_size).batch(FLAGS.batch_size, drop_remainder=FLAGS.drop_remainder)
    
    return vocab, dataset


def train(model, dataset, vocab):
    # Training
    # ==================================================
    
    unpruned_model = model(
        vocab_size = len(vocab),
        embedding_dim=FLAGS.embedding_dim,
        rnn_units=FLAGS.RNN_units,
        batch_size=FLAGS.batch_size)

    logdir = tempfile.mkdtemp()
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)]
  
    unpruned_model.compile(optimizer='adam', loss=loss)
    unpruned_model.fit(dataset, 
                       epochs = FLAGS.num_epochs, 
                       callbacks= callbacks)
    
    # Save the  model 

    _ , keras_file = tempfile.mkstemp('_unpruned.h5', dir = 'models/')
    print('Saving model to: ', keras_file)
    tf.keras.models.save_model(unpruned_model, keras_file, include_optimizer=False)
    return unpruned_model

    
def main(argv=None):
    vocab, dataset  = preprocess()
    unpruned_model = train(LSTM_model, dataset, vocab)
if __name__ == '__main__':
    tf.app.run()