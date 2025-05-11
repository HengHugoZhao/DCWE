library(reticulate)

np <- import("numpy")
pm<-function(a,b) par(mfrow=c(a,b))

words <- as.matrix(read.csv("/deac/mth/berenhautGrp/zhaoh21/word_list/word_list.txt", header = FALSE))[,1]

word_emb_fast <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/matrix/fast_emb.npy"))

XMn_fast <- word_emb_fast

vl1_fast<-apply(XMn_fast,1,function(v) sqrt(sum(v^2)))

word_emb_glove <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/matrix/glove_emb.npy"))

XMn_glove <- word_emb_glove

vl1_glove<-apply(XMn_glove,1,function(v) sqrt(sum(v^2)))


rand_mat <- matrix(rnorm(82483*300, mean = 0, sd = 1), nrow = 82483, ncol = 300)

vlR<-apply(rand_mat,1,function(v) sqrt(sum(v^2)))

pm(3,1)

hist(vl1_fast,xlim=range(c(vl1_fast, vl1_glove,vlR)),nclass=200)
hist(vl1_glove,xlim=range(c(vl1_fast, vl1_glove,vlR)),nclass=200)
hist(vlR,xlim=range(c(vl1_fast, vl1_glove,vlR)),nclass=200)

range(vl1_fast)
range(vl1_glove)
range(vlR)

euc_glove <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/matrix/euc_glove_mat.npy"))
cos_glove <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/matrix/cos_glove_mat.npy"))


D<-as.matrix(dist(rand_mat))

ans<-NULL
M<-euc_fast
for (i in 59652:nrow(euc_fast)){
 if (i%%500==0) {print(i)}
 a<-sort(M[i,])[-1]
 v<-a[1:100]
 ans<-rbind(ans,v)
 
 
}


words <- as.matrix(read.csv("/deac/mth/berenhautGrp/zhaoh21/word_list/unigram_freq.csv", header = FALSE))
words_freq <-words[-1,]
shake_words <- read.csv("/deac/mth/berenhautGrp/zhaoh21/word_list/shake_wordlist.csv")

