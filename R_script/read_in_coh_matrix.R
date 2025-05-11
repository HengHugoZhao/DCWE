library(reticulate)
np <- import("numpy")
words<-as.vector(unlist(read.csv("/deac/mth/berenhautGrp/zhaoh21/word_list/word_list.txt")))
  
  
C<-NULL
n<-length(words)
for (i in 1:n) {
  if ((i%%100)==0){print(i)}
w<-words[i]
q<-paste(c("/deac/mth/berenhautGrp/zhaoh21/coh_vec/fasttext/cohesion_vec_",w,"_fasttext.npy"),collapse="")
v<-as.vector(np$load(q))
C<-rbind(C,v)
}

rownames(C)<-words
colnames(C)<-words


coh_fast_vector <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/matrix/cohesion_matrix_fasttext.npy"))
