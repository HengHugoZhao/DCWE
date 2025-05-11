import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import os

def get_target_word_index(target_word, word_list):
    if target_word in word_list:
        index = word_list.index(target_word)
        print(f"The index of {target_word} is {index}")
    else:
        print(f"'{target_word}' is not found in the list. Exiting program.")
        sys.exit(1)
    print(f"target word indices are {index}")
    return index

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
    plt.title(f"Word {tword}  Norms vs Rank")
    plt.legend()
    
    folder_name = f"/deac/mth/berenhautGrp/zhaoh21/graphs/{type}/line_graph"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"{folder_name} created")
    
    save_name = f"word_norm_plot_{tword}_{type}"
    
    save = os.path.join(folder_name, save_name)
    
    plt.savefig(save, dpi=300)
    plt.close()
    
    print(f"fig saved as {save}")  
    
def save_word_table_with_norms(coh_words, coh_norms, cos_words, cos_norms, euc_words, euc_norms, tword, type):
    folder_name = f"/deac/mth/berenhautGrp/zhaoh21/graphs/{type}/table"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"{folder_name} created")
    
    save_name = f"word_table_{tword}_{type}.png"
    
    save = os.path.join(folder_name, save_name)
    
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
    plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Table saved as {save}")  
    
def get_tk_vec(target_word, word_list, coh_vec, cos_mat, euc_mat, norm_mat, type, k = 20):
    target_word_index = get_target_word_index(target_word, word_list)
    
    coh_tk_vec =[]
    cos_tk_vec =[]
    euc_tk_vec =[]


    coh_tk_indices = []
    cos_tk_indices = []
    euc_tk_indices = []


    coh_tk_words = []
    cos_tk_words = []
    euc_tk_words = []
    
    cos_vec = cos_mat[target_word_index,:]
    euc_vec = euc_mat[target_word_index,:]
    
    folder_name = f"/deac/mth/berenhautGrp/zhaoh21/vectors/{type}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"{folder_name} created")
    
    save_name_cos = f"cos_vec_{target_word}_{type}.npy"
    save_name_euc = f"euc_vec_{target_word}_{type}.npy"
    save_cos = os.path.join(folder_name, save_name_cos)
    save_euc = os.path.join(folder_name, save_name_euc)
    
    np.save(save_cos, cos_vec)
    np.save(save_euc, euc_vec)
    print("cos_vec, euc_vec, saved!")
    
    coh_tk_indices = np.argsort(coh_vec)[-k:][::-1]
    cos_tk_indices = np.argsort(cos_vec)[:k]
    euc_tk_indices = np.argsort(euc_vec)[:k]
    
    coh_emb = norm_mat[coh_tk_indices, :]
    cos_emb = norm_mat[cos_tk_indices, :]
    euc_emb = norm_mat[euc_tk_indices, :]
    
    coh_norm = np.linalg.norm(coh_emb, axis = 1)
    cos_norm = np.linalg.norm(cos_emb, axis = 1)
    euc_norm = np.linalg.norm(euc_emb, axis = 1)
    
    for a, b, c in zip(coh_tk_indices, cos_tk_indices, euc_tk_indices):
        coh_tk_vec.append(coh_vec[a])
        coh_tk_words.append(word_list[a])
        
        cos_tk_vec.append(cos_vec[b])
        cos_tk_words.append(word_list[b])  
        
        euc_tk_vec.append(euc_vec[c])
        euc_tk_words.append(word_list[c]) 
    
    plot_neighoring_words(coh_tk_words, coh_norm, cos_tk_words, cos_norm, euc_tk_words, euc_norm, target_word, type)

    save_word_table_with_norms(coh_tk_words, coh_norm, cos_tk_words, cos_norm, euc_tk_words, euc_norm, target_word, type)

     

    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--words", nargs="+", type=str, required=True, help="List of target words")
    parser.add_argument("-t", "--type", default = "glove", choices = ["glove", "fasttext", "normal_2d", "normal_300d","unif_2d", "unif_300d", "shake"])

    args = parser.parse_args()
    
    if args.type == "fasttext":
        euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/euc_fast_mat.npy")
        cos_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/cos_fast_mat.npy")
        norm_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/fast_emb.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/word_list.txt", header=None)
    elif args.type == "glove":
        euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/euc_glove_mat.npy")
        cos_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/cos_glove_mat.npy")
        norm_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/glove_emb.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/word_list.txt", header=None)      
    elif args.type =="normal_2d":
        euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_2d_normal_euc.npy")
        cos_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_2d_normal_cos.npy")
        norm_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_2d_normal.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/rand.txt", header=None)      
    elif args.type == "normal_300d":
        euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_300d_normal_euc.npy")
        cos_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_300d_normal_cos.npy")
        norm_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_300d_normal.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/rand.txt", header=None)      
    elif args.type == "unif_2d":
        euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_2d_unif_euc.npy")
        cos_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_2d_unif_cos.npy")
        norm_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_2d_unif.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/rand.txt", header=None)                 
    elif args.type == "unif_300d":
        euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_300d_unif_euc.npy")
        cos_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_300d_unif_cos.npy")
        norm_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_300d_unif.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/rand.txt", header=None)  
    elif args.type == "shake":
        euc_mat = np.load("/deac/mth/berenhautGrp/shake_euc.npy")
        cos_mat = np.load("/deac/mth/berenhautGrp/shake_cos.npy")
        norm_mat = np.load("/deac/mth/berenhautGrp/shak_emb.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/shake_wordlist.csv", header=None)
    
    
    print("finishing loading .....")
    type = args.type
    word_list = df[0].tolist()
    
    target_words = args.words
    
    for target_word in target_words:
        coh_vec = np.load(f"/deac/mth/berenhautGrp/zhaoh21/coh_vec/{type}/cohesion_vec_{target_word}_{type}.npy")
        print("finished loading the coh vec")
        get_tk_vec(target_word, word_list, coh_vec, cos_mat, euc_mat, norm_mat, type)
    

if __name__ == "__main__":
    main()
    