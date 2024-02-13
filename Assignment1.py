# -------------------------------------------------------------------------
# AUTHOR: Anjali Rai
# FILENAME: Assignment1.py
# SPECIFICATION: This code calculates and compares the cosine similarities between 4 documents using the numpy and sklearn library
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 2 days
# -----------------------------------------------------------*/
# Importing some Python libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Defining the documents
doc1 = "soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"
# Use the following words as terms to create your document-term matrix
# [soccer, favorite, sports, like, one, support, olympic, games]
doc1_vector = np.array([1, 1, 1, 0, 0, 0, 0, 0])  # soccer, favorite, sport
doc2_vector = np.array([1, 1, 0, 1, 1, 0, 0, 0])  # soccer, favorite, like, one
doc3_vector = np.array([1, 0, 0, 0, 0, 1, 1, 1])  # soccer, support, olympic, games
doc4_vector = np.array([1, 1, 1, 1, 0, 0, 1, 1])  # soccer, favorite, sport, like, olympic, games

# Create a matrix for all documents
doc_matrix = np.array([doc1_vector, doc2_vector, doc3_vector, doc4_vector])

# Calculate the pairwise cosine similarities
similarity_matrix = cosine_similarity(doc_matrix)

# Setting diagonal to 0 in order to ignore self-similarity detection
np.fill_diagonal(similarity_matrix, 0)

# Finding the highest similarity
max_val = np.max(similarity_matrix)
max_index = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)

# Print the highest cosine similarity following the information below
# The most similar documents are: doc1 and doc2 with cosine similarity = x
print(f"The most similar documents are: doc{max_index[0]+1} and doc{max_index[1]+1} with cosine similarity = {max_val:.4f}")

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors only
cosine_sim_12 = cosine_similarity([doc1_vector], [doc2_vector])
print("Cosine similarity between doc1 and doc2: ")
print(cosine_sim_12)

# Use cosine_similarity([X, Y, Z]) to calculate the pairwise similarities between multiple vectors
cosine_sim_123 = cosine_similarity([doc1_vector, doc2_vector, doc3_vector])
print("Pairwise cosine similarities between doc1, doc2, and doc3:")
print(cosine_sim_123)