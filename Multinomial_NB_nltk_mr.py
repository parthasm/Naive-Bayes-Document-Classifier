from __future__ import division
from nltk.corpus import movie_reviews as mr
from FilenameToCat import movie_reviews_f2c
from Evaluation import evaluation_binary
from Tokenizer import get_list_tokens_nltk_mr
from math import log
import time



def get_testset_trainset(trainToTestRatio=0.3):
    train_test = [[],[]]
    for cat in categoriesFilenameDict.keys():
        li = categoriesFilenameDict[cat]
        size=int(len(li)*trainToTestRatio)
        CatNumDocs[cat]=size
        train_test[0].extend(li[:size])
        train_test[1].extend(li[size:])
    return train_test

start_time = time.time()
size=0

##1)alternate way, to be implemented, get the categories and the corresponding
##list of files from the user,
#Here we are getting it using nltk
categoriesFilenameDict={}

#*************NON-GENERALIZED CODE***********************************


for category in mr.categories():
    categoriesFilenameDict[category]=mr.fileids(categories=category)

 

#********************************************************************
    
#This categoriesFileidDict should be obtained from the user.

    
##2)Forming b)Prepare the CatNumDocs dictionary, where the number of documents in the training set for each
             ##category are stored
    ##also forming the training set and test set
CatNumDocs={}
lis = get_testset_trainset()
trainset=lis[0]
testset=lis[1]
###--------------------DEBUG STATEMENTS----------------------
#for f in trainset:
 #   print f , FilenameCategoriesDict[f] 

#print "Freedom\n"

#for f in testset:
 #   print f , FilenameCategoriesDict[f]    
###--------------------DEBUG STATEMENTS----------------------
    


##4)Create a) a dictionary with a category as the key and dictionary of words-occurences as values
          #b) a dictionary with a category as the key and the number of words in it as the value
CatWordDict={}
CatWordCountDict={}
#val = my_dict.get(key, mydefaultval)

##5)Loop through the training set, to get the entire text from  each file
##6) Parse the string to get individual words
for fileName in trainset:
    listWords = get_list_tokens_nltk_mr(fileName)
    

##7) Check if category exists in dictionary, if not, create an empty dictionary,
    #and put word count as zero
    #and then insert words into the category's dictionary in both cases and update the word count
    cat = movie_reviews_f2c(fileName)
    CatWordDict[cat] = CatWordDict.get(cat,{})
    CatWordCountDict[cat] = CatWordCountDict.get(cat,0)
    

    CatWordCountDict[cat]+=len(listWords)
    
    for w in listWords:
        CatWordDict[cat][w] = CatWordDict[cat].get(w, 0)
        CatWordDict[cat][w]+=1
        
        

##8) Get the vocabulary length
vocabLength=0            
for dic in CatWordDict.values():
     vocabLength+=len(dic)


print "The Classifier is trained and it took"
print time.time() - start_time, "seconds"
start_time = time.time()



####Congratulations! the Classifier is trained, now it is time to run the Multinomial Naive Bayes Classifier on the test dataset
lengthTrain = len(trainset)
liResults=[]
#9) Like in the training set,Loop through the test set, to get the entire text from  each file
##10) Similar step, parse the string to get individual words
for fileName in testset:
    minimumNegLogProb=1000000000
    minCategory=''
    listWords = get_list_tokens_nltk_mr(fileName)


    
##11) Get the probability for each category,
    #can use any of the created dictionaries to wade through the categories
    for cat in  CatWordCountDict:
        #print cat , CatNumDocs[cat]/len(trainset)
        negLogProb=-log(CatNumDocs[cat]/lengthTrain)
        wordDict = CatWordDict[cat]
        countCat = CatWordCountDict[cat]
        for w in listWords:
            countWordTrain=wordDict.get(w,0)
            ratio = (countWordTrain+1)/(countCat+vocabLength)
            negLogProb-=log(ratio)           
                         
        if minimumNegLogProb>negLogProb:
            minCategory=cat
            minimumNegLogProb=negLogProb

    liResults.append((fileName,minCategory,movie_reviews_f2c(fileName)))

###--------------------DEBUG STATEMENTS----------------------
#for t in liResults:
 #   if t[1]!=t[2]:
  #      print t
###--------------------DEBUG STATEMENTS----------------------
    
###--------------------DEBUG STATEMENTS----------------------

#12) Evaluating the classifier
evaluation_binary(liResults,CatWordCountDict.keys())

print "The time taken by the trained classifier to assign labels"
print time.time() - start_time, "seconds"
