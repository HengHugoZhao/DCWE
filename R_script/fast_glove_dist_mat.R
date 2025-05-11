# Load necessary library
library(stats) # For dist()
library(words)
library(reticulate)
library(R.matlab)
library(stylo)
library(flexclust)

distCos <- function(X, Y){
  X<-rbind(X)
  x<-X[1,]
  ans <- NULL
  n <-nrow(Y)
  for (i in 1:n){
    if (i%%1000==0) {print(i)}
    y <- Y[i,]
    a <- sum(x * y)/sqrt(sum(x^2) * sum(y^2))
    ans <- c(ans, a)
  }
  names(ans)<-rownames(Y)
  ans
}



np <- import("numpy")
glove_file <- "/home/zhaoh21/kb/glove/glove.6B.300d.txt"
fastt_file <- "/deac/mth/berenhautGrp/zhaoh21/word_embeddings.txt"

lines <- readLines(glove_file, warn = FALSE)
F_lines <- readLines(fastt_file, warn = FALSE)

GM<-t(apply(cbind(lines),1,function(a) {v<-strsplit(a," ");unlist(v)}))
FM<-t(apply(cbind(F_lines),1,function(a) {v<-strsplit(a," ");unlist(v)}))

Gwords <- GM[,1]
Fwords <- FM[,1]

GM <- GM[,-1]
FM <- FM[,-1]
GM<-apply(GM,2,as.numeric)
FM<-apply(FM,2,as.numeric)

rownames(GM) <- Gwords
rownames(FM) <- Fwords

#intersection
words <- intersect(Fwords, Gwords)

GMw <- GM[is.element(Gwords, words),]
FMw <- FM[is.element(Fwords, words),]


s_GMw <- GMw[order(rownames(GMw)),]
s_FMw <- FMw[order(rownames(FMw)),]


#new vector ops
index1 = which(s_words == "king")
index2 = which(s_words == "man")
#index3 = which(s_words == "woman")
vec1 = s_GMw[index1,]
vec2 = s_GMw[index2,]
#vec3 = s_GMw[index3,]

new_vec = vec1 - vec2
new_vec <- rbind(new_vec)
euc_vec_dist <-dist2(new_vec, s_GMw, method = "euclidean")

euc_vec_output <- "euc_king_man.npy"
np$save(euc_vec_output, euc_vec_dist)

cos_vec_dis <- distCos(new_vec, s_GMw)
cos_vec_output <- "cos_king_man.npy"
np$save(cos_vec_output, cos_vec_dis)

#new vector ops
index1 = which(s_words == "king")
index2 = which(s_words == "man")
index3 = which(s_words == "woman")
vec1 = s_GMw[index1,]
vec2 = s_GMw[index2,]
vec3 = s_GMw[index3,]

new_vec = vec1 - vec2 + vec3
new_vec <- rbind(new_vec)
euc_vec_dist <-dist2(new_vec, s_GMw, method = "euclidean")

euc_vec_output <- "euc_king_man_woman.npy"
np$save(euc_vec_output, euc_vec_dist)

cos_vec_dis <- distCos(new_vec, s_GMw)
cos_vec_output <- "cos_king_man_woman.npy"
np$save(cos_vec_output, cos_vec_dis)


euc_vec_dist <- as.vector(euc_vec_dist)
cos_vec_dis <- as.vector(cos_vec_dis)

names(euc_vec_dist) <- s_words
names(cos_vec_dis) <- s_words

king_man <- cbind(names(head(sort(euc_vec_dist), 30)),
names(head(sort(-cos_vec_dis), 30)))

index1 = which(s_words == "queen")
index2 = which(s_words == "woman")
vec1 = s_GMw[index1,]
vec2 = s_GMw[index2,]

new_vec = vec1 - vec2
new_vec <- rbind(new_vec)
euc_vec_dist <-as.vector(dist2(new_vec, s_GMw, method = "euclidean"))
cos_vec_dis <- distCos(new_vec, s_GMw)

euc_vec_output <- "euc_queen_woman.npy"
np$save(euc_vec_output, euc_vec_dist)

cos_vec_output <- "cos_queen_woman.npy"
np$save(cos_vec_output, cos_vec_dis)

names(euc_vec_dist) <- s_words
names(cos_vec_dis) <- s_words
queen_woman <- cbind(names(head(sort(euc_vec_dist), 30)),
                  names(head(sort(-cos_vec_dis), 30)))

king_queen <- cbind(king_man, queen_woman)


index1 = which(s_words == "king")
index2 = which(s_words == "queen")
vec1 = rbind(s_GMw[index1,])
vec2 = rbind(s_GMw[index2,])

euc_king_dist <- as.vector(dist2(vec1, s_GMw, method = "euclidean"))
cos_king_dist <- as.vector(distCos(vec1, s_GMw))
names(euc_king_dist) <- s_words
names(cos_king_dist) <- s_words

euc_queen_dist <- as.vector(dist2(vec2, s_GMw, method = "euclidean"))
cos_queen_dist <- as.vector(distCos(vec2, s_GMw))

names(euc_queen_dist) <- s_words
names(cos_queen_dist) <- s_words

king_col <- cbind(names(head(sort(euc_king_dist), 30)),
                     names(head(sort(-cos_king_dist), 30)))
queen_col <-cbind(names(head(sort(euc_queen_dist), 30)),
                  names(head(sort(-cos_queen_dist), 30)))


index1 = which(s_words == "man")
index2 = which(s_words == "woman")
vec1 = rbind(s_GMw[index1,])
vec2 = rbind(s_GMw[index2,])

euc_man_dist <- as.vector(dist2(vec1, s_GMw, method = "euclidean"))
cos_man_dist <- as.vector(distCos(vec1, s_GMw))
names(euc_man_dist) <- s_words
names(cos_man_dist) <- s_words

euc_woman_dist <- as.vector(dist2(vec2, s_GMw, method = "euclidean"))
cos_woman_dist <- as.vector(distCos(vec2, s_GMw))

names(euc_woman_dist) <- s_words
names(cos_woman_dist) <- s_words

man_col <- cbind(names(head(sort(euc_man_dist), 30)),
                  names(head(sort(-cos_man_dist), 30)))
woman_col <-cbind(names(head(sort(euc_woman_dist), 30)),
                  names(head(sort(-cos_woman_dist), 30)))


find_data_euc <- cbind(king_col[,1], man_col[,1], king_man[,1], queen_col[,1], woman_col[,1], queen_woman[,1])
find_data_cos <- cbind(king_col[,2], man_col[,2], king_man[,2], queen_col[,2], woman_col[,2], queen_woman[,2])

write.csv(find_data_euc, "euc_king_queen.csv")
write.csv(find_data_cos, "cos_king_queen.csv")
########

s_GMw_emb <- "glove_emb.npy"
np$save(s_GMw_emb, s_GMw)

s_FMw_emb <- "fasttext_emb.npy"
np$save(s_FMw_emb, s_FMw)

s_words <- rownames(s_GMw)


wordlist_output <- "word_list.txt"
writeLines(s_words, wordlist_output)


euc_glove <-as.matrix(dist(s_GMw, method = "euclidean"))
euc_g_output <- "euc_glove_mat.npy"
np$save(euc_g_output, euc_glove)

euc_fast <-as.matrix(dist(s_FMw, method = "euclidean"))
euc_f_output <- "euc_fast_mat.npy"
np$save(euc_f_output, euc_fast)


clue_euc_g<- dist2(rbind(s_GMw[rownames(s_GMw)=="clue",]), s_GMw, method = "euclidean")[1,]
clue_euc_f <- dist2(rbind(s_FMw[rownames(s_FMw)=="clue",]), s_FMw, method = "euclidean")[1,]

clue_cos_g <- distCos(s_GMw[rownames(s_GMw)=="clue",], s_GMw)
clue_cos_f <- distCos(s_FMw[rownames(s_FMw)=="clue",], s_FMw)


head(sort(clue_euc_g), 20)
head(sort(clue_euc_f), 20)



#randon gen matrix
set.seed(123)
rand_mat <- matrix(rnorm(2000*2, mean = 0, sd = 1), nrow = 2000, ncol = 2)

rand_mat_dist <- apply(rand_mat, 1, norm, type = "2")

np$save("rand_mat_2d_normal.npy", rand_mat)

euc_rand <- as.matrix(dist(rand_mat, method="euclidean"))
rand_mat_output <- "rand_euc_2d_normal.npy"
np$save(rand_mat_output, euc_rand)

cos_rand <- as.matrix(dist.cosine(rand_mat))
np$save("rand_cos_2d_normal.npy", cos_rand)


lines <- 1:2000

# Write to a .txt file
writeLines(as.character(lines), "rand.txt")
