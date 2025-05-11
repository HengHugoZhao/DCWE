library(qdap)
library(reticulate)

np <- import("numpy")

words <- as.matrix(read.csv("/deac/mth/berenhautGrp/zhaoh21/word_list/word_list.txt", header = FALSE))[,1]
#s_words <- read.table("/deac/mth/berenhautGrp/shakespeare-words.txt", sep = ",")
word_emb <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/matrix/fast_emb.npy"))

#Swords_sr <- unlist(intersect(words, s_words))
#word_emb <- as.matrix(word_emb[is.element(words, Swords_sr),])


XMn <- word_emb

#words<- words[is.element(words, Swords_sr)]





vl1<-apply(XMn,1,function(v) sqrt(sum(v^2)))

synraw<-sapply(words,function(a) length(unlist(qdap::synonyms(a))))

synw<-sapply(words,function(a) length(intersect(words,unlist(qdap::synonyms(a)))))
synM<-t(sapply(words,function(a) {w<-unlist(qdap::synonyms(a));
q<-is.element(words,w);q*1}))
diag(synM)<-1
meanraw<-sapply(words,function(a) length(unique(qdap::synonyms(a))))
meanw<-sapply(words,function(a) {w<-unique(qdap::synonyms(a));
w2<-lapply(w,function(b) intersect(words,b));
length(w2[lapply(w2,length)!=0])
})

dataraw<-cbind(synraw,synw,meanraw,meanw,vl1)
dataraw<-data.frame(dataraw)
synonyms<-sapply(words, function(pword){
  v<-intersect(words,unlist(qdap::synonyms(pword)));print(length(v));
  v<-c(pword,v);length(v)})
meanings<-sapply(words, function(pword){v<-length(qdap::synonyms(pword));v})


i <- which(words == "solemn")

syn_words <- words[synM[i,]==1]



coh_vec <- t(as.vector(np$load("/deac/mth/berenhautGrp/zhaoh21/coh_vec/fasttext/cohesion_vec_solemn_fasttext.npy")))

ind <- which(synM[i,]==1)

dataM<-dataraw[ind,]
coh_rank<-rank(-coh_vec)[ind]

euc_vec <- (as.vector(np$load("/deac/mth/berenhautGrp/zhaoh21/vectors/fasttext/euc_vec_solemn_fasttext.npy")))
euc_rank <- rank(euc_vec)[ind]

cos_vec <- (as.vector(np$load("/deac/mth/berenhautGrp/zhaoh21/vectors/fasttext/cos_vec_solemn_fasttext.npy")))
cos_rank <- rank(cos_vec)[ind]

dataMr<-data.frame(cbind(dataM, coh_rank, cos_rank, euc_rank))



dataMr[order(dataMr$vl1),]

a<-sample(words[(synw==30)&(meanw==2)&(vl1 >= 6.5)],1);a
i <- which(words == a)
syn_words <- words[synM[i,]==1]
ind <- which(synM[i,]==1)
dataM<-data.frame(dataraw[ind,])
j<-which(rownames(dataM)==a)
rownames(dataM)[j]<-paste(c(a,"**"),collapse="")
dataM[order(dataM$vl1),]


#import the euc mat and cos mat
euc_mat <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/euc_glove_mat.npy"))
cos_mat <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/cos_glove_mat.npy"))

dword <- "animation"
euc_vec <- euc_mat[words == dword, ]
cos_vec <- cos_mat[words == dword, ]

dataMr<-data.frame(cbind(dataM, cos_rank, euc_rank))
dataMr[order(dataMr$vl1),]


ans<-dataMr
sum(ans[,6]<=ans[,7])
sum(ans[,6]>=ans[,7])
