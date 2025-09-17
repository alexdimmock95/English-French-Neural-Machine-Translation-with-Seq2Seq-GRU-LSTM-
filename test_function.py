from config import latent_dim

from preprocessing import target_features_dict, reverse_target_features_dict, max_decoder_seq_length, input_docs, encoder_input_data

from tensorflow import keras
from keras.layers import Input, LSTM, Dense 
from keras.models import load_model, Model
import numpy as np

training_model = keras.models.load_model('training_model.keras')

# Extract encoder model from trained model:
encoder_inputs = training_model.input[0]

# Get embedding layer
encoder_embedding_layer = training_model.get_layer("encoder_embedding")

# Get the actual encoder LSTM layer by name
encoder_lstm = training_model.get_layer("encoder_lstm")

# Pass through embedding before LSTM
embedded_encoder_inputs = encoder_embedding_layer(encoder_inputs)
encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(embedded_encoder_inputs)

encoder_states = [state_h_enc, state_c_enc]

encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

# Extract decoder model from trained model:
decoder_inputs = Input(shape=(1,), dtype='int32', name='dec_token_in')
decoder_embedding_layer = training_model.get_layer("decoder_embedding")
decoder_lstm = training_model.get_layer("decoder_lstm")
decoder_dense = training_model.get_layer("decoder_dense")

# New inputs for inference
decoder_state_input_hidden = Input(shape=(latent_dim,), name='decoder_h_in')
decoder_state_input_cell = Input(shape=(latent_dim,), name='decoder_c_in')
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

# Pass IDs through embedding
embedded_decoder_inputs = decoder_embedding_layer(decoder_inputs)

# Pass through LSTM
decoder_outputs, state_hidden, state_cell = decoder_lstm(embedded_decoder_inputs, initial_state=decoder_states_inputs)

# Update states
decoder_states = [state_hidden, state_cell]

# Final dense
decoder_outputs = decoder_dense(decoder_outputs)

# Inference decoder model
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_sequence(test_input):
  # Encode the input as state vectors.
  states_value = encoder_model.predict(test_input)

  # Generate empty target sequence of length 1.
  target_seq = np.array([[target_features_dict['<START>']]])
  
  # Sampling loop for a batch of sequences
  # (to simplify, here we assume a batch of size 1).
  decoded_sentence = ''

  stop_condition = False
  while not stop_condition:
    # Run the decoder model to get possible 
    # output tokens (with probabilities) & states
    output_tokens, hidden_state, cell_state = decoder_model.predict(
      [target_seq] + states_value)

    # Choose token with highest probability
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_token = reverse_target_features_dict[sampled_token_index]
    decoded_sentence += " " + sampled_token

    # Exit condition: either hit max length
    # or find stop token.
    if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
      stop_condition = True

    # Update the target sequence (of length 1).
    target_seq = np.array([[sampled_token_index]])

    # Update states
    states_value = [hidden_state, cell_state]

  return decoded_sentence

# CHANGE RANGE (NUMBER OF TEST SENTENCES TO TRANSLATE) AS YOU PLEASE
for seq_index in range(30):
  test_input = encoder_input_data[seq_index: seq_index + 1]
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