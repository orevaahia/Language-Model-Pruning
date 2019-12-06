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

from model import print_model_sparsity

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("dataset", "Data/republic_clean.txt", "Data file for the task")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 512, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("RNN units", 1024, "Dimensionality of character embedding (default: 128)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs ")


FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

