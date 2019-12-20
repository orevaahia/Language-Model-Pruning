
import numpy as np
import sys

import tensorflow as tf

from utils import load_text, split_input_target
from model import LSTM_model, pruned_lstm 

# Parameters
# ==================================================

tf.app.flags.DEFINE_string("dataset", "Data/republic_clean.txt", "Data file for the task")

tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

tf.app.flags.DEFINE_integer("embedding_dim",512,"Dimensionality of character embedding (default: 128)")
tf.app.flags.DEFINE_integer("RNN_units",512,"Dimensionality of character embedding (default: 128)")

tf.app.flags.DEFINE_string("model", "models/", "Data file for the task")
tf.app.flags.DEFINE_string("start_string", "He", "initialize text generation")
tf.app.flags.DEFINE_integer("num_generate", 1000, "Number of characters to generate ")
tf.app.flags.DEFINE_float("temperature", 1.0, "Temperature determines the randomness of the generated text")
tf.app.flags.DEFINE_boolean("drop_remainder", True, "Drop last batch if it has fewer batch size elements")
tf.app.flags.DEFINE_integer("buffer_size", 1000, "Buffer size for shuffling ")

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


def generate_text(model, start_string, vocab):
  # Evaluation step (generating text using the learned model)
  start_string = FLAGS.start_string
  model = model(vocab_size = len(vocab), 
                  embedding_dim=FLAGS.embedding_dim, 
                  rnn_units=FLAGS.RNN_units, 
                  batch_size=1)

  model.load_weights(tf.keras.models.load_model(FLAGS.model))
  model.build(tf.TensorShape([1, None]))

  # Vectorize our start string 
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Create empty string to keep our results
  generated_text = []

  # Here batch size == 1
  model.reset_states()
  for i in range(FLAGS.num_generate):
    predictions = model(input_eval)
    # remove batch dimension
    predictions = tf.squeeze(predictions, 0)

    # use a categorical distribution to predict the word returned by the model
    predictions = predictions / FLAGS.temperature 
    predicted_id = tf.random.categorical(predictions, num_samples= 1)[-1, 0].numpy()

    # Pass the predicted word as the next input into our model along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)
    generated_text.append(idxchar[predicted_id])

  return (start_string + ''.join(generated_text))


def main(argv=None):
  vocab, dataset = preprocess()
  print(generate_text(LSTM_model, 'come ', vocab))
if __name__ == '__main__':
    tf.app.run()