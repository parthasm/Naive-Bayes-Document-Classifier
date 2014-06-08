from __future__ import division
from nltk.corpus import reuters
from FilenameToCat import reuters_f2c
from Tokenizer import get_list_tokens_nltk_reuters
from Evaluation import evaluation_fraction_misclass
from Feature_Selector import mutual_information
from math import log
import time

#Forming Prepare the CatNumDocs dictionary, where the number of documents in the training set for each
             ##category are stored
    ##also forming the training set and test set

def get_testset_trainset():
    cleanFiles = [f for f in reuters.fileids() if len(reuters.categories(fileids=f))==1]    
    testset = [f for f in cleanFiles if f[:5]=='test/']
    trainset = [f for f in cleanFiles if f[:9]=='training/']
    for cat in reuters.categories():
        li=[f for f in reuters.fileids(categories=cat) if f in trainset]
        liTe = [f for f in reuters.fileids(categories=cat) if f in testset]
        if len(li)>20 and len(liTe)>20:
            CatNumDocs[cat]=len(li)
            li.extend(liTe)
            categoriesFilenameDict[cat]=li
    return [[ f for f in trainset if reuters_f2c(f) in categoriesFilenameDict],
            [ f for f in testset if reuters_f2c(f) in categoriesFilenameDict]]


start_time = time.time()

#Here, apart from the naive bayes classifier, everything is done by nltk

#2) refer the comments for the function get_testset_trainset()
categoriesFilenameDict={}
CatNumDocs={}
li = get_testset_trainset()
testset = li[1]
trainset = li[0]


#2a) Applying Feature Selection

li = mutual_information(CatNumDocs,trainset)
WordFeatures = li[0]
WordList = li[1]

#3)Create a dictionary with a word as the key and a dictionary as the value
     ## in the dictionary the category as key and number of documents in that category where it occurs as value

WordCatNumDocDict={}

#4)Loop through the reuters dataset, to get the entire text from  each file in the training set
    ## Parse the string to get individual words - done by get_list_tokens_nltk_reuters()
for fileName in trainset:
    listWords = get_list_tokens_nltk_reuters(fileName)
    cat = reuters_f2c(fileName)
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
    li = get_list_tokens_nltk_reuters(fileName)
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

    liResults.append((fileName,minCategory,reuters_f2c(fileName)))

evaluation_fraction_misclass(liResults)

print "The time taken by the trained classifier to assign labels"
print time.time() - start_time, "seconds"


