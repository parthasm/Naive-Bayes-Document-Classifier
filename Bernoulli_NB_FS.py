from __future__ import division
from FilenameToCat import f2c
from Tokenizer import get_list_tokens_nltk
import Evaluation
import Feature_Selector
import Preprocessor
from math import log
import time

#Forming Prepare the CatNumDocs dictionary, where the number of documents in the training set for each
             ##category are stored
    ##also forming the training set and test set

i = input('Enter within quotes, m for movie reviews corpus,'
          'r for reuters corpus( default is reuters) : ')
corpus=''
BinaryClassification=False
if i=='m' or i=='M':
    corpus='mr'
    BinaryClassification=True
else:
    corpus='reuters'
#Forming Prepare the CatNumDocs dictionary, where the number of documents in the training set for each
             ##category are stored
    ##also forming the training set and test set

start_time = time.time()

#Here, apart from the naive bayes classifier, everything is done by nltk

#2) refer the comments for the function get_testset_trainset()

li = Preprocessor.get_testset_trainset(corpus)
testset = li[1]
trainset = li[0]
li = Preprocessor.startup()
categoriesFilenameDict=li[0]
CatNumDocs = li[1]


#2a) Applying Feature Selection

i = input('Enter within quotes, g for gini coefficient,'
          'm for mutual information( default is mutual information)  as feature selector: ')

if i=='g' or i=='G':
    li = Feature_Selector.gini(trainset,corpus)    
else:
    li = Feature_Selector.mutual_information(trainset,corpus)    
WordFeatures = li[0]
WordList = li[1]

#3)Create a dictionary with a word as the key and a dictionary as the value
     ## in the dictionary the category as key and number of documents in that category where it occurs as value

WordCatNumDocDict={}

#4)Loop through the dataset, to get the entire text from  each file in the training set
    ## Parse the string to get individual words - done by get_list_tokens_nltk()
for fileName in trainset:
    listWords = get_list_tokens_nltk(corpus,fileName)
    cat = f2c(corpus,fileName)
    listWords = [w for w in listWords if WordFeatures[cat].get(w,-100000)!=-100000]
    
    for w in set(listWords):
       WordCatNumDocDict[w]=WordCatNumDocDict.get(w,{})
       WordCatNumDocDict[w][cat]=WordCatNumDocDict[w].get(cat,0)
       WordCatNumDocDict[w][cat]+=1

for w in WordCatNumDocDict:
    for cat in CatNumDocs:
        Nct = WordCatNumDocDict[w].get(cat,0)
        ratio = (Nct+1)/(CatNumDocs[cat]+2)
        WordCatNumDocDict[w][cat]=ratio


####Congratulations! the Classifier is trained, now it is time to run the Multinomial Naive Bayes Classifier on the test dataset
print "The Classifier is trained and it took"
print time.time() - start_time, "seconds"
start_time = time.time()


liResults=[]
#5) Like in the training set,Loop through the test set, to get the individual words
for fileName in testset:
    minimumNegLogProb=1000000000
    minCategory=''
    li = get_list_tokens_nltk(corpus,fileName)
    setListWords = set([w for w in li if w in WordList])
    
    
##6) Get the probability for each category,
    #using the CatNumDocs dictionary to wade through the categories
    for cat in  CatNumDocs:
        negLogProb=-log(CatNumDocs[cat]/len(trainset))
        for w in WordCatNumDocDict:
            if w in setListWords:
                negLogProb-=log(WordCatNumDocDict[w][cat])
            else:
                negLogProb-=log(1-WordCatNumDocDict[w][cat])
                         
        if minimumNegLogProb>negLogProb:
            minCategory=cat
            minimumNegLogProb=negLogProb

    liResults.append((fileName,minCategory,f2c(corpus,fileName)))
if BinaryClassification:
    Evaluation.evaluation_binary(liResults)
else:
   Evaluation.evaluation_multi_class(liResults,CatNumDocs.keys())

print "The time taken by the trained classifier to assign labels"
print time.time() - start_time, "seconds"


