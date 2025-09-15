import numpy as np
import re
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
from test_function import decode_sequence

# Create a function to translate a new sentence
