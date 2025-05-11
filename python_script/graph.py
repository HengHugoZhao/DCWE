import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def get_target_word_indices(target_word_list, word_list):
    target_word_indices = []
    for target_word in target_word_list:
        if target_word in word_list:
            index = word_list.index(target_word)
            print(f"The index of {target_word} is {index}")
            target_word_indices.append(index)
        else:
            print(f"'{target_word}' is not found in the list. Exiting program.")
            sys.exit(1)
    print(f"target word indices are {target_word_indices}")
    return target_word_indices

def append_matrix(mat, vec):
    length = len(vec)
    new_mat = np.zeros((length + 1, length + 1))
    new_mat[:length, :length] = mat
    new_mat[length, :length] = vec
    new_mat[:length, length] = vec
    
    return new_mat

def append_cos_matrix(mat, vec):
    length = len(vec)
    new_mat = np.ones((length + 1, length + 1)) *2
    new_mat[:length, :length] = mat
    new_mat[length, :length] = vec
    new_mat[:length, length] = vec
    
    return new_mat

def plot_neighoring_words(coh_words, coh_norm, cos_words, cos_norm, euc_words, euc_norm, tword, type):
# Rank is just the index of the word in the list
    plt.figure(figsize=(12, 6))
    ranks = np.arange(1, len(coh_words) + 1)  # [1, 2, 3, 4]
    
    plt.plot(ranks, coh_norm, color='red', linestyle='-', marker='o', label="Cohesion Distance")
    for i, word in enumerate(coh_words):
        plt.text(ranks[i], coh_norm[i], word, fontsize=6, color='red', ha='left')

    # Plot cosine distance (black)
    plt.plot(ranks, cos_norm, color='black', linestyle='-', marker='o', label="Cosine Distance")
    for i, word in enumerate(cos_words):
        plt.text(ranks[i], cos_norm[i], word, fontsize=6, color='black', ha='right')
   
    # Plot euclidean distance (blue)
    plt.plot(ranks, euc_norm, color='blue', linestyle='-', marker='o', label="Euclidean Distance")
    for i, word in enumerate(euc_words):
        plt.text(ranks[i], euc_norm[i], word, fontsize=6, color='blue', ha='left') 

    # Labels & Legend
    plt.xlabel("Rank")
    plt.ylabel("Norm")
    plt.title("Word Norms vs Rank")
    plt.legend()

    save = f"/deac/mth/berenhautGrp/zhaoh21/graphs/word_norm_plot_{tword}_{type}"
    plt.savefig(save)
    plt.close()
    
    print("plt saved")

def save_word_table_with_norms(coh_words, coh_norms, cos_words, cos_norms, euc_words, euc_norms, tword, type):
    filename=f"/deac/mth/berenhautGrp/zhaoh21/graphs/word_table_{tword}_{type}.png"
    coh_norms = np.round(coh_norms, 3)
    cos_norms = np.round(cos_norms, 3)
    euc_norms = np.round(euc_norms, 3)
    # Create a DataFrame with words and their corresponding 2-norms
    df = pd.DataFrame({
        "Cohesion Words": coh_words,
        "Cohesion 2-Norm": coh_norms,
        "Cosine Words": cos_words,
        "Cosine 2-Norm": cos_norms,
        "Euclidean Words": euc_words,
        "Euclidean 2-Norm": euc_norms
    })
    
    # Set figure size dynamically based on number of rows
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5))  

    # Hide axes
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns, 
                     cellLoc='center', 
                     loc='center')

    # Save the table as a figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Table saved as {filename}")

type = "fasttext"

# coh_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/coh_vec/cohesion_['bulk', 'censure']_glove.npy")
# euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/euc_{type}_mat.npy")
# cos_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/cos_{type}_mat.npy")
# norm_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/{type}_emb.npy")
coh_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/coh_vec/cohesion_['aggression', 'condemnation']_fasttext.npy")
euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/euc_fast_mat.npy")
cos_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/cos_fast_mat.npy")
norm_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/fast_emb.npy")
df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list.txt", header=None)
word_list = df[0].tolist()

target_word = ["condemnation"]
k = 20
ana = False
j=1

target_word_indice =  get_target_word_indices(target_word, word_list)

coh_vec = coh_mat[j,:]
# print("shape of coh: ", coh_mat.shape)
# print("shape of euc: ", euc_mat.shape)
# print("shape of cos: ", cos_mat.shape)
# print("shape of nrom: ", norm_mat.shape)
# print(euc_mat.shape)


# new_vec_name = "king-man+woman"

# df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list.txt",header=None)


if ana == True:
    new_euc_vec = np.load("/deac/mth/berenhautGrp/zhaoh21/alg_vec/euc_king_man_woman.npy")
    new_euc_vec = np.array(new_euc_vec)
    new_euc_vec = new_euc_vec[0]
    new_euc_mat = append_matrix(euc_mat, new_euc_vec)
    euc_mat = new_euc_mat
    
    new_cos_vec = np.load("/deac/mth/berenhautGrp/zhaoh21/alg_vec/cos_king_man_woman.npy")
    new_cos_vec = np.array(new_cos_vec)
    # print(new_cos_vec)
    new_cos_mat = append_cos_matrix(cos_mat, new_cos_vec)
    cos_mat = new_cos_mat
    
    new_norm_vec = np.load("/deac/mth/berenhautGrp/zhaoh21/alg_vec/king_man_woman.npy")
    new_norm_vec = np.array(new_norm_vec)
    new_norm_vec = new_norm_vec[0]
    new_norm_mat = np.vstack([norm_mat, new_norm_vec])
    norm_mat = new_norm_mat
    print(norm_mat.shape)
    
    
    word_list.append(new_vec_name)



# target_word_indices = 1513

target_word_indices = target_word_indice[0]
word = word_list[target_word_indices]



coh_tk_vec =[]
cos_tk_vec =[]
euc_tk_vec =[]


coh_tk_indices = []
cos_tk_indices = []
euc_tk_indices = []


coh_tk_words = []
cos_tk_words = []
euc_tk_words = []




cos_vec = cos_mat[target_word_indices,:]
euc_vec = euc_mat[target_word_indices,:]

# print("coh_vec shape", coh_vec.shape)
# print("cos_vec shape", cos_vec.shape)
np.save(f"/deac/mth/berenhautGrp/zhaoh21/vectors/cos_vec_{target_word_indices}_{type}.npy", cos_vec)
np.save(f"/deac/mth/berenhautGrp/zhaoh21/vectors/euc_vec_{target_word_indices}_{type}.npy", euc_vec)
print("cos_vec, euc_vec, saved!")

coh_tk_indices = np.argsort(coh_vec)[-k:][::-1]
cos_tk_indices = np.argsort(cos_vec)[:k]
euc_tk_indices = np.argsort(euc_vec)[:k]

# print("coh_tk_indices", len(coh_tk_indices))
# print("cos_tk_indices", len(cos_tk_indices))

coh_emb = norm_mat[coh_tk_indices, :]
cos_emb = norm_mat[cos_tk_indices, :]
euc_emb = norm_mat[euc_tk_indices, :]

# print("cos_emb: ", cos_emb.shape)


coh_norm = np.linalg.norm(coh_emb, axis = 1)
cos_norm = np.linalg.norm(cos_emb, axis = 1)
euc_norm = np.linalg.norm(euc_emb, axis = 1)
# print("cos_ebd: ", cos_norm.shape)

# exit(1)


for a in coh_tk_indices:
    coh_tk_vec.append(coh_vec[a])
    coh_tk_words.append(word_list[a])
    
for b in cos_tk_indices:
    cos_tk_vec.append(cos_vec[b])
    cos_tk_words.append(word_list[b])

print("len of cos words ", len(cos_tk_words))

for c in euc_tk_indices: 
    euc_tk_vec.append(euc_vec[c])
    euc_tk_words.append(word_list[c])
    
# print("coh")
# print(coh_tk_words)
# print(coh_norm)
# print("shape of words,", coh_tk_words)
# print("shape of norm,", coh_norm)

plot_neighoring_words(coh_tk_words, coh_norm, cos_tk_words, cos_norm, euc_tk_words, euc_norm, word, type)

save_word_table_with_norms(coh_tk_words, coh_norm, cos_tk_words, cos_norm, euc_tk_words, euc_norm, word, type)


# indices = np.argsort(mat)

# new_list=[]

# for i in indices:
#     new_list.append(word_list[i])
    
# for i in range(100):
#     print(new_list[i])
    
