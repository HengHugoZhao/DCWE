import numpy as np
import matplotlib.pyplot as plt

coh_D = np.load(f"/deac/mth/berenhautGrp/zhaoh21/matrix/cohesion_matrix_fasttext.npy")
coh_D = 1 - coh_D

plt.hist(coh_D, ) # bins specifies the number of bins
plt.xlabel("cosine distance")
plt.ylabel("Frequency")
plt.title("Histogram Example")
plt.savefig("/deac/mth/berenhautGrp/zhaoh21/graphs", dpi=450, bbox_inches='tight')