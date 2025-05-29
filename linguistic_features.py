#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag, FreqDist
from collections import Counter
import string

# Make sure you have these NLTK resources downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample text (replace this with your own input later)
text = """
The hotel was absolutely fantastic! The room was clean, the staff were friendly, and the location was perfect.
However, the Wi-Fi was slow, and breakfast was a bit disappointing. Overall, I would recommend it.
"""

# Preprocess text
sentences = sent_tokenize(text)
words = word_tokenize(text.lower())
words = [w for w in words if w.isalpha()]  # Remove punctuation/numbers

# Basic stats
num_sentences = len(sentences)
num_words = len(words)
unique_words = set(words)
lexical_diversity = len(unique_words) / num_words

# Sentence and word length
avg_sentence_length = num_words / num_sentences
avg_word_length = sum(len(w) for w in words) / num_words

# POS tagging
pos_tags = pos_tag(words)
pos_counts = Counter(tag for word, tag in pos_tags)

# Most common words
fdist = FreqDist(words)
most_common_words = fdist.most_common(10)

# Output results
print("---- Linguistic Features ----")
print(f"Number of sentences: {num_sentences}")
print(f"Number of words: {num_words}")
print(f"Number of unique words: {len(unique_words)}")
print(f"Lexical diversity: {lexical_diversity:.2f}")
print(f"Average sentence length (in words): {avg_sentence_length:.2f}")
print(f"Average word length: {avg_word_length:.2f}")
print("\nPart of Speech Counts:")
for tag, count in pos_counts.items():
    print(f"{tag}: {count}")
print("\nTop 10 most common words:")
for word, freq in most_common_words:
    print(f"{word}: {freq}")

