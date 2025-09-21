import numpy as np
import re

if __name__ == "__main__":
  print('\nPreprocessing script running...\n')

# Import language dataset
data_path = "/Users/Alex/Documents/Coding/2. Data Scientist - Natural Language Processing Specialist/deep_learning_project/language_pair/fra-eng/fra.txt"

# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:   # Open in read mode
  lines = f.read().split('\n')    # Split each line by newline

# Building empty lists to hold sentences
input_docs = []   # Initialise list for input docs (English)
target_docs = []    # Initialise list for target docs (French)

# Building empty vocabulary sets
input_tokens = set()    # Initialise set for input tokens 
target_tokens = set()   # Initialise set for target tokens

# Adjust number of lines to preprocess
for line in lines[:2000]:    # For each line in first N lines
  # Input and target sentences are separated by tabs
  input_doc, target_doc = line.split('\t')[:2]    # Splits each line at tab, takes index 0 and 1 and returns as input_doc and target_doc
  
  # Clean and tokenise text
  input_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", input_doc))   # Joins all tokens (word items or punctuation items) in input_doc with spaces. Apostrophe included in word items eg "don't".
  target_doc = "START " + " ".join(re.findall(r"[\w]+|[^\s\w]", target_doc)) + " END"    # Joins all tokens (word items or punctuation items) in target_doc with spaces. Apostrophe not included in word items eg "n'est" becomes "n" "est"

  # Appending each input sentence to docs lists
  input_docs.append(input_doc)    # Add input doc to input docs list
  target_docs.append(target_doc)    # Add target doc to target docs list

  # Tokenise sentences and build vocab sets
  for token in re.findall(r"[\w']+|[^\s\w]", input_doc):    # Find all word or punctuation items in input_doc, add as tokens in input_tokens

    if token not in input_tokens:
      input_tokens.add(token)   # Add token to input_tokens if not already in set
  
  for token in re.findall(r"[\w]+|[^\s\w]", target_doc):   # Find all word or punctuation items in target_doc, add as tokens in target_tokens
  
    if token not in target_tokens:
      target_tokens.add(token)    # Add token to target_tokens if not already in set

# Sort tokens
input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

# Create num_encoder_tokens and num_decoder_tokens on set lengths
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

# Calculate max sequence lengths from input and target docs
max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w]+|[^\s\w]", target_doc)) for target_doc in target_docs])

# Print max sequence lengths
if __name__ == "__main__":
  print(f'Max encoder seq length: {max_encoder_seq_length} \nMax decoder seq length: {max_decoder_seq_length} \n')

# Create token-index dictionaries for input and target languages
input_features_dict = dict(
    [(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict(
    [(token, i) for i, token in enumerate(target_tokens)])

# Create reverse token-index dictionaries (index-token) for input and target languages
reverse_input_features_dict = dict(
    (i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict(
    (i, token) for token, i in target_features_dict.items())

# Create 2D array of zeros for encoder input, decoder input and decoder target data. Dimensions: (number of lines inputted above, max sequence length for input or target language)
encoder_input_data = np.zeros((len(input_docs), max_encoder_seq_length), dtype="int32")   # Will include values for each token in each input sequence, 0s after
decoder_input_data = np.zeros((len(input_docs), max_decoder_seq_length), dtype="int32")   # Will include "START " before each input sequence, 0s after
decoder_target_data = np.zeros((len(input_docs), max_decoder_seq_length), dtype="int32")   # Will include 0s before each input sequence, " END" after

# Populate 2D arrays with index, value pairs from language doc and features dict
for i, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):    # For each index within both the zipped list of input_docs and target_docs
    
    for t, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):    # For each subsequent (index, value) pair in the tokenised input_doc
        encoder_input_data[i, t] = input_features_dict[token]   # For sentence i, at position t, store the token index of that word from input_features_dict

    for t, token in enumerate(re.findall(r"[\w]+|[^\s\w]", target_doc)):    # For each subsequent (index, value) pair in the tokenised target_doc
        decoder_input_data[i, t] = target_features_dict[token]    # For sentence i, at position t, store the token index of that word from target_features_dict
        if t > 0:
            decoder_target_data[i, t - 1] = target_features_dict[token]   # Push decoder_target_value t value to the left by 1, so that it is ahead of decoder_input_value by one timestep

# Print number of input and target tokens
def print_tokens():
  print("Num input tokens:", len(input_features_dict))  # Print total number input tokens
  print("Num target tokens:", len(target_features_dict))    # Print total number target tokens
  print("Num reverse input tokens:", len(reverse_input_features_dict))    # Print total number reverse input tokens
  print("Num reverse target tokens:", len(reverse_target_features_dict))    # Print total number reverse target tokens

  print("\nFirst 50 input tokens:", list(input_features_dict.keys())[:50])    # Print first 50 input tokens
  print("\nFirst 50 target tokens:", list(target_features_dict.keys())[:50])    # Print first 50 target tokens

  # The number of keys in each dictionary should match the number of tokens (input_features_dict, num_encoder_tokens), (target_features_dict, num_decoder_tokens), (reverse_input_features_dict, num_encoder_tokens), (reverse_target_features_dict, num_decoder_tokens)
  if len(input_features_dict) != num_encoder_tokens:
    print("\nError: Num encoder tokens does not match the length of input_features_dict")
  if len(target_features_dict) != num_decoder_tokens:
    print("\nError: Num decoder tokens does not match the length of target_features_dict")
  if len(reverse_input_features_dict) != num_encoder_tokens:
    print("\nError: Num reverse input tokens does not match the length of reverse_input_features_dict")
  if len(reverse_target_features_dict) != num_decoder_tokens:
    print("\nError: Num reverse target tokens does not match the length of reverse_target_features_dict")

# Call the function to print tokens if this script is run (not imported as a module)
if __name__ == "__main__":
  print_tokens()

if __name__ == "__main__":
  print('Preprocessing script finished.\n')