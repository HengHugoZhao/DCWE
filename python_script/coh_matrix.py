import os 
import pandas as pd
import numpy as np

# Paths
folder_path = '/deac/mth/berenhautGrp/zhaoh21/coh_vec/glove'  # adjust your folder path
wordlist_path = '/deac/mth/berenhautGrp/zhaoh21/word_list/word_list.txt'  # adjust to your actual wordlist path

# Read wordlist
words = pd.read_csv(wordlist_path, header=None).squeeze().tolist()

# Build filenames from words
file_list = [f"cohesion_vec_{word}_glove.npy" for word in words]

# Load vectors and stack into matrix
vectors = []
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    vector = np.load(file_path)
    vectors.append(vector)

# Stack into a matrix
matrix = np.vstack(vectors)

np.save('/deac/mth/berenhautGrp/zhaoh21/matrix/cohesion_matrix_glove.npy', matrix)

print("Matrix shape:", matrix.shape)
