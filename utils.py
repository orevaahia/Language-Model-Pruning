import numpy as np 
import string

def load_text(data):
    """
    Loads data file, then decode for py2 compat
    """
    text = open(data, 'rb').read().decode(encoding='utf-8')
    print ('Length of text: {} characters'.format(len(text)))
    return text

def clean_text(text):
    """
    Cleans text data 
    """
    text = text.replace('--', '')
    text_tokens = text.split()
    table = str.maketrans('', '', string.punctuation)
    text_tokens = [word.translate(table) for word in text_tokens]
    text_tokens = [word for word in text_tokens if word.isalpha()]
    text_tokens = [word.lower() for word in text_tokens]
    return text_tokens

def split_input_target(chunk):
    """
    
    """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
