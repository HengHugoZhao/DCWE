import numpy as np
import pandas as pd
import sys
import time
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def randomly_select_words(word_list, num_words):
    """
    Randomly select a specified number of words from a list.
    
    Args:
        word_list (list): List of words to select from.
        num_words (int): Number of words to select.
        
    Returns:
        list: List of randomly selected words.
    """
    if num_words > len(word_list):
        raise ValueError("Number of words to select exceeds the size of the word list.")
    
    selected_words = np.random.choice(word_list, num_words, replace=False)
    return selected_words.tolist()


def gen_matrix(rows, cols, min_val=1, max_val = 5):
    matrix = np.random.uniform(min_val, max_val, size=(rows, cols))
    return matrix

def append_matrix(mat, vec):
    length = len(vec)
    new_mat = np.zeros((length + 1, length + 1))
    new_mat[:length, :length] = mat
    new_mat[length, :length] = vec
    new_mat[:length, length] = vec
    
    return new_mat

def get_target_word_indices(target_word_list, word_list):
    target_word_indices = []
    for target_word in target_word_list:
        if target_word in word_list:
            index = word_list.index(target_word)
            print(f"The index of {target_word} is {index}")
            target_word_indices.append(index)
        else:
            print(f"{target_word} is not found in the list. Exiting program.")
            sys.exit(1)
    print(f"target word indices are {target_word_indices}")
    return target_word_indices
    
def get_target_word_index(taget_word, word_list):
    if taget_word in word_list:
        index = word_list.index(taget_word)
        print(f"The index of '{taget_word}' is: {index}")
        return index
    else:
        print(f"'{taget_word}' is not found in the list. Exiting program.")
        sys.exit(1)  # Exit with a non-zero status to indicate an error

def cal_coh_for_multi_target_words(dis_mat, word_list, target_word_indices, type):
    n=len(dis_mat)
    
    folder_name = f"/deac/mth/berenhautGrp/zhaoh21/coh_vec/{type}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"{folder_name} created")
    
    # m=len(target_word_indices)
    # matrix = np.zeros((0,n))
    # print(f"the original matrix shape is {matrix.shape}")
    for target_word_index in target_word_indices:
        target_word = word_list[target_word_index]
        save_name = f"cohesion_vec_{target_word}_{type}.npy"
        save = os.path.join(folder_name, save_name) 
        
        print(f"Calculating distance for #{target_word}# target words against {n} filtered GloVe words")
        
        row = np.zeros(n)
        row_sum = np.zeros(n)
        
        for y in range(n):
            uxy = 0
            print(f"number y is: {y}")
            if target_word_index == y:
                continue
            dxy = dis_mat[target_word_index, y]
            
            for v in range(n):
                dxv = dis_mat[target_word_index, v]
                dyv = dis_mat[y, v]
                
                if dxv <= dxy or dyv <= dxy:
                    uxy+=1
                    if dxv < dyv:
                        row[v] = 1
                    elif dxv == dyv:
                        row[v] = 0.5
                    else:
                        row[v] = 0
            row = row/uxy
            row_sum = row_sum + row
            row_sum= np.array(row_sum)
        np.save(save, row_sum)
        print(f"coh_vec saved at {save}")
        # matrix = np.vstack([matrix, row_sum])
        
        
def pald_cohesion_vector(d,xs, word_list, type):
    folder_name = f"/deac/mth/berenhautGrp/zhaoh21/coh_vec/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"{folder_name} created")
        
    n = d.shape[0]
    for x in xs:
        save_name = f"cohesion_vec_{word_list[x]}_{type}.npy"
        save = os.path.join(folder_name, save_name) 
        row = np.zeros(n)
        for y in tqdm(range(n), desc=f"Comparing to {word_list[x]}"):
            if y == x:
                continue
            dx = d[:, x]
            dy = d[:, y]
            dxy = d[y, x]
            
            uxy_mask = (dx <= dxy) | (dy <= dxy)
            uxy = np.where(uxy_mask)[0]
            
            # print(f"number y is: {y}")
            # print(f"word y is {word_list[y]} ")
            # # print("uxy: ", uxy)
            # print(f"len of uxy is {len(uxy)}")
            uxy_word_list = []
            if len(uxy) <= 20:
                support_x_word_list = [word_list[i] for i in uxy if dx[i] < dy[i]]
                support_y_word_list = [word_list[i] for i in uxy if dy[i] < dx[i]]
                
                string = f"number y is: {y} | word y is {word_list[y]} | len of uxy is {len(uxy)}| uxy word list is {uxy_word_list} | supporting x word list is {support_x_word_list} | supporting y word list is {support_y_word_list}"
                
                file_path = f"/deac/mth/berenhautGrp/zhaoh21/log/coh_vec_log_{word_list[x]}.txt"
                if not os.path.exists(file_path):
                    with open(file_path, "w") as f:
                        f.write(string + "\n")  # Create an empty file
                else:
                    with open(file_path, "a") as f:
                        f.write(string + "\n")
            
            if len(uxy) == 0:
                continue
            
            wx = (dx[uxy] < dy[uxy]).astype(float) + 0.5 * (dx[uxy] == dy[uxy])
            row[uxy] += (1 / len(uxy)) * wx
            
        row /= (n-1)
        # np.save(save, row)
        print(f"coh_vec saved at {save}")
        
        
    
    
def main(): 
    parser = argparse.ArgumentParser()
    # parser.add_argument("-w", "--words", nargs="+", type=str, required=True, help="List of target words")
    parser.add_argument("-t", "--type", default="glove", choices=["glove", "fasttext", "normal_2d", "normal_300d","unif_2d", "unif_300d", "shake"])

    args = parser.parse_args()
           
    start = time.time()
    type = args.type
    # ana = False
    top_word_list = pd.read_csv("/deac/mth/berenhautGrp/top2000freqwords.csv", header=None)    
    top_word_list = [str(w).strip() for w in top_word_list[0].tolist()]

    target_word_list = randomly_select_words(top_word_list, 20)

    print(target_word_list)

    if args.type == "fasttext":
        euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/euc_fast_mat.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/word_list.txt", header=None)
    elif args.type == "glove":
        euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/euc_glove_mat.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/word_list.txt", header=None)      
    elif args.type =="normal_2d":
        euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_2d_normal_euc.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/rand.txt", header=None)      
    elif args.type == "normal_300d":
        euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_300d_normal_euc.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/rand.txt", header=None)      
    elif args.type == "unif_2d":
        euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_2d_unif_euc.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/rand.txt", header=None)                 
    elif args.type == "unif_300d":
        euc_mat = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/rand_mat_300d_unif_euc.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/rand.txt", header=None)
    elif args.type == "shake":
        euc_mat = np.load("/deac/mth/berenhautGrp/shake_euc.npy")
        df = pd.read_csv("/deac/mth/berenhautGrp/zhaoh21/word_list/shake_wordlist.csv", header=None)  

        # cos_mat = np.load("/deac/mth/berenhautGrp/zhaoh21/cos_glove_mat.npy")
    # mat = np.load("euclidean_distance_matrix.npy")
    word_list = [str(w).strip() for w in df[0].tolist()]
    print(f"first word {word_list[0]}")

    # print("euc_mat: ", euc_mat[:5, :5])



    # if ana == True:
    #     new_euc_vec = np.load("/deac/mth/berenhautGrp/zhaoh21/alg_vec/euc_king_man_woman.npy")
    #     new_euc_vec = np.array(new_euc_vec)
    #     new_euc_vec = new_euc_vec[0]
    #     new_euc_mat = append_matrix(euc_mat, new_euc_vec)

    #     word_list.append(new_vec_name)

    #     target_word_indices = get_target_word_indices(target_word_list, word_list)
    #     coh_mat = cal_coh_for_multi_target_words(new_euc_mat, word_list, target_word_indices)
    #     save_name = f'/deac/mth/berenhautGrp/zhaoh21/coh_vec/cohesion_{new_vec_name}.npy'

    # else:
    #     # target_word_indices = get_target_word_index(target_word, word_list)
    #     target_word_indices = get_target_word_indices(target_word_list, word_list)
    #     coh_mat = cal_coh_for_multi_target_words(euc_mat, word_list, target_word_indices)

        # cohesion = cal_coh_for_single_target_words(mat, word_list, 0)
    target_word_indices = get_target_word_indices(target_word_list, word_list)
    # cal_coh_for_multi_target_words(euc_mat, word_list, target_word_indices, args.type)
    pald_cohesion_vector(euc_mat, target_word_indices, word_list, args.type)
    # Create the directory if it doesn't exist
    # Save the cohesion matrix as a .npy file
    end = time.time()
    print("The time of execution of above program is :",(end-start) / 60, "minutes")
    
if __name__ == "__main__":
    main()


