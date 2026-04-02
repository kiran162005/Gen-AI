import numpy as np
import pandas as pd
from collections import defaultdict

#Sample corpus
text = """
1. Text Input and Data Collection
Data Collection: Gathering text data from various sources such as websites, books, social media or proprietary databases.
Data Storage: Storing the collected text data in a structured format, such as a database or a collection of documents.
2. Text Preprocessing
Preprocessing is crucial to clean and prepare the raw text data for analysis. Common preprocessing steps include:

Tokenization: Splitting text into smaller units like words or sentences.
Lowercasing: Converting all text to lowercase to ensure uniformity.
Stopword Removal: Removing common words that do not contribute significant meaning, such as "and," "the," "is."
Punctuation Removal: Removing punctuation marks.
Stemming and Lemmatization: Reducing words to their base or root forms. Stemming cuts off suffixes, while lemmatization considers the context and converts words to their meaningful base form.
Text Normalization: Standardizing text format, including correcting spelling errors, expanding contractions and handling special characters.
3. Text Representation
Bag of Words (BoW): Representing text as a collection of words, ignoring grammar and word order but keeping track of word frequency.
Term Frequency-Inverse Document Frequency (TF-IDF): A statistic that reflects the importance of a word in a document relative to a collection of documents.
Word Embeddings: Using dense vector representations of words where semantically similar words are closer together in the vector space (e.g., Word2Vec, GloVe).
"""
tokens = text.lower().split()

window_size = 2

#Build vocabulary
vocab = sorted(set(tokens))
word_to_id = {w:i for i, w in enumerate(vocab)}
id_to_word = {i: w for w, i in word_to_id.items()}

#Initialize co-occurrence matrix
co_matrix = np.zeros((len(vocab), len(vocab)), dtype=int)

# Build co-occurrence counts
for i, word in enumerate(tokens):
    word_id = word_to_id[word]

    start = max(i - window_size, 0)
    end = min(i + window_size + 1, len(tokens))

    for j in range(start, end):
        if i != j:
            context_word = tokens[j]
            context_id = word_to_id[context_word]
            co_matrix[word_id][context_id] += 1

# Convert to table
df = pd.DataFrame(co_matrix, index=vocab, columns=vocab)

print("Co-occurrence Matrix (Window size = 2):")
print(df)
