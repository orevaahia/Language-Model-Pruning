import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity

#end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs

pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=2000,
                                                   end_step= 6000,
                                                   frequency=100)
}

def GRU_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """
    Function to build the GRU model 
    Args:
        vocab_size: Length of the vocabulary in chars
        embedding_dim: The embedding dimension
        rnn_units: Number of RNN units
        batch_size: Batch size
    Returns:
        GRU model 

    """
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
     tf.keras.layers.Dense(vocab_size)
  ])
    return model

def LSTM_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """
    Function that builds the  LSTM model 
    Args:
        vocab_size: Length of the vocabulary in chars
        embedding_dim: The embedding dimension
        rnn_units: Number of RNN units
        batch_size: Batch size
    Returns:
        LSTM model 
    """
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True),
    tf.keras.layers.Dense(vocab_size)
   ])
    return model
def pruned_lstm(vocab_size, embedding_dim, rnn_units, batch_size):
  """
  
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

def print_model_sparsity(pruned_model):
  """Prints sparsity for the pruned layers in the model.
  Model Sparsity Summary
  --
  prune_lstm_1: (kernel, 0.5), (recurrent_kernel, 0.6)
  prune_dense_1: (kernel, 0.5)
  Args:
    pruned_model: keras model to summarize.
  Returns:
    None
  """
  def _get_sparsity(weights):
    return 1.0 - np.count_nonzero(weights) / float(weights.size)

  print("Model Sparsity Summary ({})".format(pruned_model.name))
  print("--")
  for layer in pruned_model.layers:
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      prunable_weights = layer.layer.get_prunable_weights()
      if prunable_weights:
        print("{}: {}".format(
            layer.name, ", ".join([
                "({}, {})".format(weight.name,
                                  str(_get_sparsity(K.get_value(weight))))
                for weight in prunable_weights
            ])))
  print("\n")
