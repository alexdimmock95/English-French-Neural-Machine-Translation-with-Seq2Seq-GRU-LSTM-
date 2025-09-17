from config import latent_dim, embedding_dim, batch_size, epochs

from preprocessing import num_encoder_tokens, num_decoder_tokens, decoder_target_data, encoder_input_data, decoder_input_data

import matplotlib.pyplot as plt

# Add Dense to the imported layers
from keras.layers import Input, LSTM, Dense, Embedding
from keras.models import Model
from keras.optimizers import Adam

# UNCOMMENT THE TWO LINES BELOW IF YOU ARE GETTING ERRORS ON A MAC
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def build_training_model(num_encoder_tokens, num_decoder_tokens, latent_dim, embedding_dim):
    """
    Builds and compiles a seq2seq model for training.
    
    Args:
        num_encoder_tokens (int): Number of features in the encoder input.
        num_decoder_tokens (int): Number of features in the decoder output.
        latent_dim (int): Dimensionality of LSTM hidden states.
    
    Returns:
        model (Model): Compiled seq2seq training model.
        encoder_lstm (LSTM): Encoder LSTM layer (needed for inference if you rebuild encoder_model).
        decoder_lstm (LSTM): Decoder LSTM layer (needed for inference).
        decoder_dense (Dense): Dense output layer for the decoder.
    """
    # ---------- Encoder ----------
    encoder_inputs = Input(shape=(None, ), name='encoder_input')
    encoder_embedding = Embedding(input_dim=num_encoder_tokens, 
                                  output_dim=embedding_dim, 
                                  mask_zero=True, 
                                  name='encoder_embedding')(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')
    _, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # ---------- Decoder ----------
    decoder_inputs = Input(shape=(None, ), name='decoder_input')
    decoder_embedding = Embedding(input_dim=num_decoder_tokens, 
                                  output_dim=embedding_dim, 
                                  mask_zero=True, 
                                  name='decoder_embedding')(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    # ---------- Full Seq2Seq Model ----------
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model, encoder_lstm, decoder_lstm, decoder_dense

# Build and train
training_model, encoder_lstm, decoder_lstm, decoder_dense = \
    build_training_model(num_encoder_tokens, num_decoder_tokens, latent_dim, embedding_dim)

# Train the model:
print(f'\nTraining the model:\n')
history = training_model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
)

# Print model summary
print('\nModel summary:\n')
training_model.summary()

# Save model
training_model.save('training_model.keras')
print('\nModel saved to training_model.keras')

# Plot training history values
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# To Add:
# Bleu Score evaluation