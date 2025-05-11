library(pald)
library(reticulate)
library(stylo)

np <- import("numpy")

set.seed(123)
rand_mat <- matrix(rnorm(7*2, mean = 0, sd = 1), nrow = 7, ncol = 2)
np$save("rand_mat_2d_normal.npy", rand_mat)

euc_rand <- as.matrix(dist(rand_mat, method="euclidean"))
np$save("rand_mat_2d_normal_euc.npy", euc_rand)

cos_rand <- as.matrix(dist.cosine(rand_mat))
np$save("rand_mat_2d_normal_cos.npy", cos_rand)

coh_2d_normal <- pald(euc_rand, show_plot = FALSE)$C

coh_2d_normal_vec <- coh_2d_normal[1,]

my_coh_vec <- np$load("/deac/mth/berenhautGrp/zhaoh21/coh_vec/normal_2d/cohesion_vec_1_normal_2d.npy")
my_coh_vec2 <- np$load("/deac/mth/berenhautGrp/zhaoh21/coh_vec/normal_2d/cohesion_vec_2_normal_2d.npy")
my_coh_vec3 <- np$load("/deac/mth/berenhautGrp/zhaoh21/coh_vec/normal_2d/cohesion_vec_3_normal_2d.npy")
my_coh_vec4 <- np$load("/deac/mth/berenhautGrp/zhaoh21/coh_vec/normal_2d/cohesion_vec_4_normal_2d.npy")
my_coh_vec5 <- np$load("/deac/mth/berenhautGrp/zhaoh21/coh_vec/normal_2d/cohesion_vec_5_normal_2d.npy")
my_coh_vec6 <- np$load("/deac/mth/berenhautGrp/zhaoh21/coh_vec/normal_2d/cohesion_vec_6_normal_2d.npy")
my_coh_vec7 <- np$load("/deac/mth/berenhautGrp/zhaoh21/coh_vec/normal_2d/cohesion_vec_7_normal_2d.npy")


my_coh_vec <- np$load("/deac/mth/berenhautGrp/zhaoh21/coh_vec/shake/cohesion_vec_vision_shake.npy")



