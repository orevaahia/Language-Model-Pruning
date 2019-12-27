# Imports
import numpy as np
import sys
import tempfile

import tensorflow as tf 
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing import sequence

from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

# Build a pruned model layer by layer
from tensorflow_model_optimization.sparsity import keras as sparsity

from utils import load_text, split_input_target, loss
from model import GRU_model, print_model_sparsity

# Parameters
# ==================================================

# Data loading params
tf.app.flags.DEFINE_string("dataset", "Data/republic_clean.txt", "Data file for the task")

# Model Hyperparameters
tf.app.flags.DEFINE_integer("embedding_dim", 512, "Dimensionality of character embedding ")
tf.app.flags.DEFINE_integer("RNN_units", 1024, "Dimensionality of character embedding ")

# Training parameters
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs ")
tf.app.flags.DEFINE_integer("buffer_size", 1000, "Buffer size for shuffling ")
tf.app.flags.DEFINE_boolean("drop_remainder", True, "Drop last batch if it has fewer batch size elements")

#Pruning parameters
tf.app.flags.DEFINE_float("initial_sparsity", 0.00, "Sparsity at which pruning begins")
tf.app.flags.DEFINE_float("final_sparsity", 0.50, " Sparsity at which pruning ends")
tf.app.flags.DEFINE_integer("begin_step", 2000, " Step at which to begin pruning")
tf.app.flags.DEFINE_integer("end_step", 8000, " Step at which to end pruning")



FLAGS = tf.app.flags.FLAGS
FLAGS(sys.argv)

pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=FLAGS.initial_sparsity,
                                                   final_sparsity=FLAGS.final_sparsity,
                                                   begin_step=FLAGS.begin_step,
                                                   end_step= FLAGS.end_step,
                                                   frequency=200)
}
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

def pruned_lstm(vocab_size, embedding_dim, rnn_units, batch_size):
  """
  Function that builds the pruned LSTM model using the pruning parameters
    Args:
        vocab_size: Length of the vocabulary in chars
        embedding_dim: The embedding dimension
        rnn_units: Number of RNN units
        batch_size: Batch size
    Returns:
        LSTM model 
  """

  pruned_model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
      sparsity.prune_low_magnitude(
          tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True), **pruning_params),
      sparsity.prune_low_magnitude(
          tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True), **pruning_params),
      sparsity.prune_low_magnitude(
          tf.keras.layers.Dense(vocab_size), **pruning_params)
      
  ])
  
  return pruned_model


def train_pruned_model(model, dataset, vocab):
    pruned_model = model(
        vocab_size = len(vocab),
        embedding_dim=FLAGS.embedding_dim,
        rnn_units=FLAGS.RNN_units,
        batch_size=FLAGS.batch_size)
    
    logdir = tempfile.mkdtemp()
    callbacks=[sparsity.UpdatePruningStep(),
                                sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)]

    pruned_model.compile(optimizer='adam', loss=loss)
    pruned_model.fit(dataset,
                     epochs=FLAGS.num_epochs, 
                     callbacks= callbacks )
    
    # Save the pruned model for size comparison later
    _, checkpoint_file = tempfile.mkstemp(str(FLAGS.final_sparsity) + '_pruned.h5', dir = 'models/')
    print('Saving pruned model to: ', checkpoint_file)
    tf.keras.models.save_model(pruned_model, checkpoint_file, include_optimizer=False)

    # Strip the pruning wrappers from the pruned model as it is only needed for training
    final_pruned_model = sparsity.strip_pruning(pruned_model)

    _, pruned_keras_file = tempfile.mkstemp('_final_pruned.h5', dir = 'models/')
    print('Saving pruned model to: ', pruned_keras_file)
    tf.keras.models.save_model(final_pruned_model, pruned_keras_file, include_optimizer=False)

    return pruned_model, final_pruned_model
    

def main(argv=None):
    vocab, dataset = preprocess()
    pruned_model, final_pruned_model = train_pruned_model(pruned_lstm, dataset, vocab)
if __name__ == '__main__':
    tf.app.run()