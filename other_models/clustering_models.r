# ===========================
# ========== Set up =========
# ===========================
library(cluster)
library(tm)



model_folder = '/Users/Leonova/Repos/jd-classifier/other_models'
setwd(model_folder)


# ===========================
# ========= Load Data =======
# ===========================
#d_tm = read.csv("jd_dense_dtm.csv")





#corpus = Corpus(dir)
# myCorpus = read.csv("jd_corpus.csv")
myCorpus = read.csv("corpus.csv")

# Set the ids to specific column
corpus = VCorpus(DataframeSource(myCorpus), readerControl = 
                   list(reader = readTabular(mapping = list(content = "description", id = "title"))))



#corpus <- Corpus(VectorSource(myCorpus$description))
#dtm = TermDocumentMatrix(corpus)



ndocs <- length(corpus)
# ignore extremely rare words i.e. terms that appear in less then 1% of the documents
minTermFreq <- ndocs * 0.01
# ignore overly common words i.e. terms that appear in more than 50% of the documents
maxTermFreq <- ndocs * .5
dtm = DocumentTermMatrix(corpus,
                         control = list(
                           stopwords = TRUE, 
                           wordLengths=c(4, 15),
                           removePunctuation = T,
                           removeNumbers = T,
                           #stemming = T,
                           bounds = list(global = c(minTermFreq, maxTermFreq))
                         ))

# Summary
dtm


m  <- as.matrix(dtm)
distMatrix <- dist(m, method="euclidean")

groups <- hclust(distMatrix, method="ward.D")
plot(groups, cex=0.9, hang=-1)
rect.hclust(groups, k=5)






library(tm)
library(proxy)
library(RTextTools)
library(fpc)   
library(wordcloud)
library(cluster)
library(tm)
library(stringi)
library(proxy)
library(wordcloud)










s






ndocs <- length(corpus)
# ignore extremely rare words i.e. terms that appear in less then 1% of the documents
minTermFreq <- ndocs * 0.01
# ignore overly common words i.e. terms that appear in more than 50% of the documents
maxTermFreq <- ndocs * .5


dtm = DocumentTermMatrix(corpus,
                         control = list(
                           stopwords = TRUE, 
                           wordLengths=c(4, 15),
                           removePunctuation = T,
                           removeNumbers = T,
                           #stemming = T,
                           bounds = list(global = c(minTermFreq, maxTermFreq))
                         ))
#dtm <- dtm[, names(head(sort(colSums(as.matrix(dtm))), 400))]
#dtm <- dtm[, names(sort(colSums(as.matrix(dtm))))]
#print(as.matrix(dtm))
write.csv((as.matrix(dtm)), "test.csv")
#head(sort(as.matrix(dtm)[18,], decreasing = TRUE), n=15)
dtm.matrix = as.matrix(dtm)
#wordcloud(colnames(dtm.matrix), dtm.matrix[28, ], max.words = 20)



