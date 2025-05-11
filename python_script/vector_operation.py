import csv
import numpy as np

def filter_lines_by_len(file_path):
    """
    Reads a file and prints lines where 'len of uxy' is less than 30.

    Args:
        file_path (str): Path to the file to be read.
    """
    with open(file_path, 'r') as file:
        for line in file:
            # Extract the 'len of uxy' value from the line
            if "len of uxy is" in line:
                try:
                    len_of_uxy = int(line.split("len of uxy is")[1].strip())
                    if len_of_uxy < 30:
                        print(line.strip())
                except ValueError:
                    # Skip lines where parsing fails
                    continue

# Example usage
file_path = '/deac/mth/berenhautGrp/zhaoh21/log/coh_vec_log.txt'  # Update with the correct path
filter_lines_by_len(file_path)