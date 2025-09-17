import numpy as np
import re

# Importing our translations
# for example: "spa.txt" or "spa-eng/spa.txt"
data_path = "/Users/Alex/Documents/Coding/2. Data Scientist - Natural Language Processing Specialist/deep_learning_project/language_pair/fra-eng/fra.txt"

# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:
  lines = f.read().split('\n')

# Building empty lists to hold sentences
input_docs = []
target_docs = []
# Building empty vocabulary sets
input_tokens = set()
target_tokens = set()

# Adjust the number of lines so that
# preprocessing doesn't take too long for you
for line in lines[:2000]:
  # Input and target sentences are separated by tabs
  input_doc, target_doc = line.split('\t')[:2]
  # Appending each input sentence to input_docs
  input_docs.append(input_doc)

  target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
  # Redefine target_doc below
  # and append it to target_docs:
  target_doc = '<START> ' + target_doc + ' <END>'
  target_docs.append(target_doc)

  # Now we split up each sentence into words
  # and add each unique word to our vocabulary set
  for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
    # print(token)
    # Add your code here:
    if token not in input_tokens:
      input_tokens.add(token)
  for token in target_doc.split():
    # print(token)
    # And here:
    if token not in target_tokens:
      target_tokens.add(token)

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

# Create num_encoder_tokens and num_decoder_tokens:
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])

input_features_dict = dict(
    [(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict(
    [(token, i) for i, token in enumerate(target_tokens)])

reverse_input_features_dict = dict(
    (i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict(
    (i, token) for token, i in target_features_dict.items())

encoder_input_data = np.zeros((len(input_docs), max_encoder_seq_length), dtype="int32")
decoder_input_data = np.zeros((len(input_docs), max_decoder_seq_length), dtype="int32")
decoder_target_data = np.zeros((len(input_docs), max_decoder_seq_length), dtype="int32")

for i, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
    for t, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
        encoder_input_data[i, t] = input_features_dict[token]
    for t, token in enumerate(target_doc.split()):
        decoder_input_data[i, t] = target_features_dict[token]
        if t > 0:
            decoder_target_data[i, t - 1] = target_features_dict[token]


# Print number of input and target tokens
def print_tokens():
  print("Num input tokens:", len(input_features_dict))
  print("Num target tokens:", len(target_features_dict))
  print("Num reverse input tokens:", len(reverse_input_features_dict))
  print("Num reverse target tokens:", len(reverse_target_features_dict))

  print("\nFirst 50 input tokens:", list(input_features_dict.keys())[:50])
  print("\nFirst 50 target tokens:", list(target_features_dict.keys())[:50])

  if len(input_features_dict) != num_encoder_tokens:
    print("\nError: Num encoder tokens does not match the length of input_features_dict")
  if len(target_features_dict) != num_decoder_tokens:
    print("\nError: Num decoder tokens does not match the length of target_features_dict")
  if len(reverse_input_features_dict) != num_encoder_tokens:
    print("\nError: Num reverse input tokens does not match the length of reverse_input_features_dict")
  if len(reverse_target_features_dict) != num_decoder_tokens:
    print("\nError: Num reverse target tokens does not match the length of reverse_target_features_dict")

if __name__ == "__main__":
    print_tokens()