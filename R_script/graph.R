library(qdap)
library(reticulate)
pm<-function(a,b) par(mfrow=c(a,b))

np <- import("numpy")

words <- as.matrix(read.csv("/deac/mth/berenhautGrp/zhaoh21/word_list/word_list.txt", header = FALSE))[,1]

word_emb <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/matrix/fast_emb.npy"))

word_emb<-read.csv(file="glove_emb.csv",row.names=1)
words<-read.csv(file="glove_words.csv",row.names=1)
words<-as.vector(words)[[1]]
XMn <- word_emb
P<-prcomp(XMn)$x
vl1<-apply(XMn,1,function(v) sqrt(sum(v^2)))










getpic<-function(a, pr=FALSE,pp=TRUE){
  
  print(length(intersect(words,unlist(qdap::synonyms(a)))))
  w<-unique(qdap::synonyms(a))
  w2<-lapply(w,function(b) intersect(words,b));
  w2<-w2[lapply(w2,length)!=0]
  
  
  col_vec<-NULL
  word_vec<-NULL
  for (i in 1:length(w2)){
    q<-length(w2[[i]])
    col_vec<-c(col_vec,rep(i,q))
    word_vec<-c(word_vec,w2[[i]])}
  
  col_vec<-c(1,col_vec)
  word_vec<-c(a,word_vec)
  
  
  q<-which(is.element(word_vec,words))
  word_vec<-word_vec[q]
  col_vec<-col_vec[q]
  
  cex_vec<-sapply(word_vec,function(a) vl1[words==a])
  
  if (pr) {PM<-P} else {PM<-XMn}
  U<-NULL
  for (i in 1:length(word_vec)){
    U<-rbind(U,PM[words==word_vec[i],])}
  #rownames(U)<-word_vec
  
  cols<-rainbow(max(col_vec))
  
  word_vec[word_vec==a]<-"*"
  cex2<-cex_vec
  names(cex2)<-word_vec
  
  cex_vec<-cex_vec/max(vl1)*3
  col_vec<-cols[col_vec]
  col_vec[1]<-"black"
  
  if (pr) {UM<-U} else {
    #U<-rbind(U,apply(U,2,mean))
    UM<-prcomp(U)$x
    #col_vec<-c(col_vec,"purple")
    #word_vec<-c(word_vec,"+")
    #cex_vec<-c(cex_vec,2*max(cex_vec))
    cex_vec[1]<-2*max(cex_vec)
  }
  D<-as.matrix(dist(UM));d<-apply(D,1,sum);
  d<-D[1,]
  names(d)<-word_vec
  #cex2<-c(cex2,sqrt(sum(apply(U,2,mean)^2)))
  plot(UM[,1:2],type="n",asp=1,xlim=range(UM[,1]),ylim=range(UM[,2]))
  text(UM[,1:2],labels=word_vec,col=col_vec,cex=cex_vec)
  title(a)
  if(pp){
    cex_vec<-1.2*cex_vec
    cex_vec[1]<-1.5*cex_vec[1]
    dev.new()
    pald(as.matrix(dist(U)),vertex.label.color=col_vec,vertex.size=.1,
         only_strong=TRUE,vertex.label=word_vec,edge.width=.5,layout=UM[,1:2],vertex.label.cex=cex_vec)}
  title(a)
  dz<-unique(cbind(names(d),d))
  d<-as.numeric(dz[,2])
  names(d)<-rownames(dz)
  round(sort(d),3)
}





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


i <- which(words == "volume")

syn_words <- words[synM[i,]==1]



coh_M <- t(as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/coh_vec/cohesion_['hanker', 'volume']_glove.npy")))

coh_vec <- coh_M[,2]

ind <- which(synM[i,]==1)

dataM<-dataraw[ind,]
coh_rank<-rank(-coh_vec)[ind]

euc_vec <- (as.vector(np$load("/deac/mth/berenhautGrp/zhaoh21/vectors/euc_vec_79960_glove.npy")))
euc_rank <- rank(euc_vec)[ind]

cos_vec <- (as.vector(np$load("/deac/mth/berenhautGrp/zhaoh21/vectors/cos_vec_79960_glove.npy")))
cos_rank <- rank(cos_vec)[ind]

dataMr<-data.frame(cbind(dataM, coh_rank, cos_rank, euc_rank))



dataMr[order(dataMr$vl1),]

a<-sample(words[synw==15],1);a
i <- which(words == a)
syn_words <- words[synM[i,]==1]
ind <- which(synM[i,]==1)
dataM<-data.frame(dataraw[ind,])
j<-which(rownames(dataM)==a)
rownames(dataM)[j]<-paste(c(a,"**"),collapse="")
dataM[order(dataM$vl1),]








fast_embd <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/matrix/fast_emb.npy"))
vl1<-apply(fast_embd,1,function(v) sqrt(sum(v^2)))
cos_matrix_fast <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/matrix/cos_fast_mat.npy"))
euc_matrix_fast <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/matrix/euc_fast_mat.npy"))

M<-NULL
for(i in seq(1,4,0.01)){
  v <- i^300
  x <- sum(vl1<= i)
  M<-rbind(M,c(v,x))  
}
M2<-M
M2[,1]<-log(M2[,1])
plot(M2[1:6,], pch=16,type="l")

plot(seq(1,4,.01)[1:10],M2[1:10,1]/M2[1:10,2], pch=16,type="l")



hist(cos_matrix_fast[row(cos_matrix_fast) > col(cos_matrix_fast)], nclass=100)


A <- cos_matrix_fast[1:1000, 1:1000]
hist(A[row(A) > col(A)], nclass=100)

B <- euc_matrix_fast[1:1000, 1:1000]
hist(B[row(B)>col(B)], nclass=100)

par(mfrow=c(2,1))
