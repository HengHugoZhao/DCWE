library(qdap)
library(reticulate)
library(igraph);library(pald)

np <- import("numpy")

words <- as.matrix(read.csv("/deac/mth/berenhautGrp/zhaoh21/word_list/word_list.txt", header = FALSE))[,1]

word_emb <- as.matrix(np$load("/deac/mth/berenhautGrp/fast_emb.npy"))

Swords<-read.table("/deac/mth/berenhautGrp/shakespeare-words.txt",sep=",")
Swordsr<-unlist(intersect(words,Swords))

words_freq <- as.matrix(read.csv("/deac/mth/berenhautGrp/zhaoh21/word_list/unigram_freq.csv", header = FALSE))[-1,]
Fwords<-words_freq[,1]
freq<-as.numeric(words_freq[,2])
Swordsr<-unlist(intersect(Swordsr,Fwords))
#write.csv(cbind(Swordsr),"shake_words.csv")

word_emb<- as.matrix(word_emb[is.element(words,Swordsr),])
words<-words[is.element(words,Swordsr)]

Swordsr_freq <- sapply(Swordsr, function(a) freq[Fwords == a])
#np$save("shak_emb.npy", word_emb)

XMn <- word_emb

vl1<-apply(XMn,1,function(v) sqrt(sum(v^2)))
syns<-TRUE
if (syns) {
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
}

dataraw<-cbind(synraw,synw,meanraw,meanw,vl1, Swordsr_freq)
dataraw<-data.frame(dataraw)

###hugo's way
a<-sample(words[(synraw <= 4)&(meanraw <= 4)&(rank(Swordsr_freq) >= 1000)],1);a




####

dr<-dataraw[order(vl1),]
drr<-dataraw[order(-vl1),]
#cummed<-function(v){n<-length(v);sapply(1:n,function(u){median(v[1:u])})}
cumgr<-function(v){n<-length(v);sapply(1:n,function(u){sum(v[1:u]>0)})}

bigs<-max(synraw)
bigm<-max(meanraw)
us<-unlist(sapply(0:bigs,function(a) mean(sapply(words[synraw==a],nchar))))
um<-unlist(sapply(0:bigm,function(a) mean(sapply(words[meanraw==a],nchar))))
usv<-unlist(sapply(0:bigs,function(a) mean(vl1[synraw==a])))
uv<-unlist(sapply(0:bigm,function(a) mean(vl1[meanraw==a])))
pm<-function(a,b) par(mfrow=c(a,b))
pm(2,2)
plot(0:bigs,us, type="l",xlab="number of synonyms",
     ylab="avg number of characters");points(0:bigs,us,pch=16,col="red")
plot(0:bigs,usv, type="l", xlab="number of synonyms",
     ylab="avg norm");points(0:bigs,usv,pch=16,col="blue")
plot(0:bigm,um, type="l",xlab="number of meanings",
     ylab="avg number of characters");points(0:bigm,um,pch=16,col="red")
plot(0:bigm,uv, type="l",xlab="number of meanings",
     ylab="avg norm");points(0:bigm,uv,pch=16,col="blue")

table(sapply(rownames(dataraw[(synraw>100),]), nchar))



plot(cumsum((dr)$synraw)/(1:length(words)),type="l",col="blue",ylab="avg number of synonyms")
lines(cumsum((drr)$synraw)/(1:length(words)),col="red")

cdr<-cumgr((dr)$synraw)
cdrr<-cumgr((drr)$synraw)


#plot(cdr/(1:length(words)),type="l",col="blue",ylab="avg number of synonyms")
#lines(cdrr/(1:length(words)),col="red")

plot(cdr,type="l",col="blue",ylab="number of words with at least one synonym")
lines(cdrr,col="red")


#Q<-synM;Q<-pmax(Q,t(Q))
#diag(Q)<-0;r<-apply(Q,1,sum)
#Q<-Q[r>0,r>0]

#Q<-pmax(Q,t(Q))
#g<-graph.adjacency(Q,mode="undirected")


#n<-8000;plot(drr[1:n,]$vl1,drr[1:n,]$synraw)

synonyms<-sapply(words, function(pword){
  v<-intersect(words,unlist(qdap::synonyms(pword)));print(length(v));
  v<-c(pword,v);length(v)})
meanings<-sapply(words, function(pword){v<-length(qdap::synonyms(pword));v})




#coh_M <- t(as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/coh_vec/cohesion_['bulk', 'censure']_glove.npy")))

#coh_vec <- coh_M[,2]

DE<-as.matrix(dist(XMn))
C<-cohesion_matrix(dist(XMn))
DC<-as.matrix(stylo::dist.cosine(XMn))
Thr<-.5*mean(diag(C))
rownames(C)<-Swordsr;colnames(C)<-Swordsr;
rownames(DE)<-Swordsr;colnames(DE)<-Swordsr;
rownames(DC)<-Swordsr;colnames(DC)<-Swordsr;

np$save("shake_euc.npy", DE)
np$save("shake_coh.npy", C)
np$save("shake_cos.npy", DC)
####

ADE<-t(apply(DE,1,rank))
ADC<-t(apply(DC,1,rank))
AC<-t(apply(-C,1,rank))
n<-nrow(AC)

j<-3*(synw+1);
j<-5
vDE<-apply(ADE,2,function(v) sum(v<=j))
vDC<-apply(ADC,2,function(v) sum(v<=j))
vC<-apply(AC,2,function(v) sum(v<=j))
pm(2,3)
hist((vDE+1),breaks=seq(-2,max((vDE+1))+1,1))
hist((vDC+1),breaks=seq(-2,max((vDC+1))+1,1))
hist((vC+1),breaks=seq(-2,max((vC+1))+1,1))
hist((j),breaks=seq(-2,max((j))+1,1))
hist((3*meanw),breaks=seq(-2,max((3*meanw+1))+1,1))


bigs<-max(vDC)
bigm<-max(vC)
us<-unlist(sapply(0:bigs,function(a) mean(sapply(words[vDC==a],nchar))))
um<-unlist(sapply(0:bigm,function(a) mean(sapply(words[vC==a],nchar))))
usv<-unlist(sapply(0:bigs,function(a) mean(synw[vDC==a])))
uv<-unlist(sapply(0:bigm,function(a) mean(synw[vC==a])))
pm<-function(a,b) par(mfrow=c(a,b))
pm(2,2)
plot(0:bigs,us, type="l",xlab="number of synonyms (cosine)",
     ylab="avg number of characters");points(0:bigs,us,pch=16,col="red")
plot(0:bigs,usv, type="l", xlab="number of synonyms (cosine)",
     ylab="avg number of synonyms");points(0:bigs,usv,pch=16,col="blue")
plot(0:bigm,um, type="l",xlab="number of synonyms (cohesion)",
     ylab="avg number of characters");points(0:bigm,um,pch=16,col="red")
plot(0:bigm,uv, type="l",xlab="number of synonyms (cohesion)",
     ylab="avg number of synonyms");points(0:bigm,uv,pch=16,col="blue");points(vC,synw,cex=.5,pch=16)




k<-.01
j<-quantile(DE,c(k));j;vDE<-apply(DE,2,function(v) sum(v<=j))
j<-quantile(DC,c(k));j;vDC<-apply(DC,2,function(v) sum(v<=j))
j<-quantile(-C,c(k));
#j<-(-1)*Thr;
j;vC<-apply(-C,2,function(v) sum(v<=j))
pm(2,2)
hist((vDE+1),breaks=seq(-2,max((vDE+1))+1,1));tail(rev(table(vDE)),20)
hist((vDC+1),breaks=seq(-2,max((vDC+1))+1,1));tail(rev(table(vDC)),20)
hist((vC+1),breaks=seq(-2,max((vC+1))+1,1));tail(rev(table(vC)),20)
hist((synw+1),breaks=seq(-2,max((synw+1))+1,1));tail(rev(table(synw)),20)
rev(table(vDE))
rev(table(vDC))

get_array<-function(a){
  i<-which(words==a);i
  ind <- which(synM[i,]==1);length(ind)
  
  dataM<-dataraw[ind,]
  
  coh_vec <- C[i,]
  coh_rank<-rank(-coh_vec)[ind]
  
  #euc_vec <- (as.vector(np$load("/deac/mth/berenhautGrp/zhaoh21/vectors/euc_vec_11246_glove.npy")))
  euc_vec<-DE[i,]
  euc_rank <- rank(euc_vec)[ind]
  
  #cos_vec <- (as.vector(np$load("/deac/mth/berenhautGrp/zhaoh21/vectors/cos_vec_11246_glove.npy")))
  cos_vec <- DC[i,]
  cos_rank <- rank(cos_vec)[ind]
  
  dataMr<-data.frame(cbind(dataM, coh_rank, cos_rank, euc_rank))
  
  sum(coh_vec>=Thr)
  #q<-Swordsr[coh_vec>=Thr]
  dataMr[order(dataMr$vl1),]
}

#####
a<-sample(words[(synw==30)&(meanw==2)&(vl1 >= 6.5)],1);a
i <- which(words == a)
syn_words <- words[synM[i,]==1]
ind <- which(synM[i,]==1)
dataM<-data.frame(dataraw[ind,])
j<-which(rownames(dataM)==a)
rownames(dataM)[j]<-paste(c(a,"**"),collapse="")
dataM[order(dataM$vl1),]


#import the euc mat and cos mat
euc_mat <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/euc_fast_mat.npy"))
cos_mat <- as.matrix(np$load("/deac/mth/berenhautGrp/zhaoh21/cos_fast_mat.npy"))

dword <- "animation"
euc_vec <- euc_mat[words == dword, ]
cos_vec <- cos_mat[words == dword, ]

dataMr<-data.frame(cbind(dataM, cos_rank, euc_rank))
dataMr[order(dataMr$vl1),]

ans<-NULL;k<-NULL
for (i in 1:length(Swordsr)){
  A<-get_array(Swordsr[i])
  if (nrow(A)>2){
    q1<-A$coh_rank
    q2<-A$cos_rank
    a<-c(length(q1),sum(q1<=q2),sum(q2<=q1)/length(q1),sum(q1<=q2)/length(q1),
         as.vector(dataraw[i,]))
    ans<-rbind(ans,a)
    k<-c(k,Swordsr[i])
    rownames(ans)<-k
  }
}
ans<-data.frame(ans);ans<-apply(ans,2,unlist)
ans2<-ans[(ans[,3]==1)&(ans[,1]>3),]
ans2<-ans2[order(unlist(ans2[,1])),]

ans5<-ans[(ans[,4]>.8)&(ans[,1]>3),]
tail(ans5[order(unlist(ans5[,1])),],20)


n<-ncol(XMn)
j<-n/20
