from config import latent_dim        # Paramaters defined in config.py

from preprocessing import target_features_dict, reverse_target_features_dict, max_decoder_seq_length, input_docs, encoder_input_data

from tensorflow import keras
from keras.layers import Input, LSTM, Dense 
from keras.models import load_model, Model
import numpy as np

# Load model from training file
training_model = keras.models.load_model('training_model.keras')

# ---------- Encoder model for inference ----------

# Extract encoder input layer from trained model
encoder_inputs = training_model.input[0]

# Extract encoder embedding layer from trained model
encoder_embedding_layer = training_model.get_layer("encoder_embedding")

# Pass encoder_inputs through encoder_embedding to get dense vector for each timestep
embedded_encoder_inputs = encoder_embedding_layer(encoder_inputs)

# Extract the encoder LSTM layer from trained model
encoder_lstm = training_model.get_layer("encoder_lstm")

# Save progressive results and final states for LSLTM on embedded encoder inputs
encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(embedded_encoder_inputs)

# Save hidden and cell states in a list
encoder_states = [state_h_enc, state_c_enc]

# Create inference encoder model
# Inputs:
#   encoder_inputs: source sequence (token IDs), shape (batch_size, encoder_timesteps)
# Outputs:
#   state_h_enc: encoder's final hidden state, shape (batch_size, latent_dim)
#   state_c_enc: encoder's final cell state, shape (batch_size, latent_dim)
encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

# ---------- Decoder model for inference ----------

# Extract decoder model from trained model:
decoder_inputs = Input(shape=(1,), dtype='int32', name='dec_token_in') # shape(1,) ensures passing one token at a time

# Extract embedding layer from trained model
decoder_embedding_layer = training_model.get_layer("decoder_embedding") # Embedding weights used during training are used here

# Extract the decoder LSTM from trained model
decoder_lstm = training_model.get_layer("decoder_lstm") # LSTM is from training model

# Extract the decoder Dense layer from trained model
decoder_dense = training_model.get_layer("decoder_dense") # Dense probability calcs from training model

# New inputs for inference
  # decoder_state_input_hidden: placeholder for decoder's previous hidden state (h) at current timestep
  # decoder_state_input_cell: placeholder for decoder's previous cell state (c) at current timestep
  # decoder_states_inputs: list combining hidden and cell state inputs
decoder_state_input_hidden = Input(shape=(latent_dim,), name='decoder_h_in')
decoder_state_input_cell = Input(shape=(latent_dim,), name='decoder_c_in')
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

# Pass IDs through embedding
embedded_decoder_inputs = decoder_embedding_layer(decoder_inputs) # Turn integer token ID into dense vector 

# Pass through LSTM
decoder_outputs, state_hidden, state_cell = decoder_lstm(embedded_decoder_inputs, initial_state=decoder_states_inputs) # Feeds in one token and last states, returns next softmax distribution, updated states

# Update states
decoder_states = [state_hidden, state_cell] # These will be carried forward into the next timestep

# Final dense
decoder_outputs = decoder_dense(decoder_outputs) # Converts LSTM output into vocab probabilities

# Inference decoder model
decoder_model = Model(
  [decoder_inputs] + decoder_states_inputs, # Input: current token + decoder_states_inputs
  [decoder_outputs] + decoder_states # Output: softmax over vocab + decoder states
)

def decode_sequence(test_input):
  '''
  Description:
  Arguments: test_input: sequence from input dataset
  Outputs: decoded_sentence: predicted output sequence in target dataset
  '''
  # Encode the input as state h and c vectors. Result of calling .predict() will give final hidden and cell states for that input.
  states_value = encoder_model.predict(test_input)

  # Generate empty target sequence of length 1.
  target_seq = np.array([[target_features_dict['START']]])
  
  # Sampling loop for a batch of sequences
  # (to simplify, here we assume a batch of size 1)
  decoded_sentence = ''

  stop_condition = False

  while not stop_condition:
    
    # Run the decoder model
    # Inputs:
      # [target_seq]:the last token
      # states_value: the last hidden and cell states
    # Outputs: 
      # output_tokens: softmax probabilities for target token
      # hidden_state: short-term memory saved from last token
      # cell_state: long-term memory saved from last token
    output_tokens, hidden_state, cell_state = decoder_model.predict(
      [target_seq] + states_value)

    # Choose token with highest probability
    sampled_token_index = np.argmax(output_tokens[0, -1, :]) # Greedy encoding grabbing the highest probability from the vocab list dimension at first index and last timestep
    sampled_token = reverse_target_features_dict[sampled_token_index] # Turns the index into a target word
    decoded_sentence += " " + sampled_token # Adds the word to decoded_sentence

    # Exit condition: either hit max length
    # or find stop token.
    if (sampled_token == 'END' or len(decoded_sentence) > max_decoder_seq_length):
      stop_condition = True

    # Update the target sequence (of length 1) with the latest token index
    target_seq = np.array([[sampled_token_index]])

    # Update states
    states_value = [hidden_state, cell_state]

  return decoded_sentence

# Change range to number of sentences to translate
for seq_index in range(5):
  test_input = encoder_input_data[seq_index: seq_index + 1]   # Slice to ensure returning a batch with shape (1, length)
  decoded_sentence = decode_sequence(test_input)
  print('-')
  print('Input sentence:', input_docs[seq_index])
  print('Decoded sentence:', decoded_sentence)

#################----------------------####################
'''
Next steps:
- Add attention mechanism
- Byte pair encoding or entence piece for OOVs
'''