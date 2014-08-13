from __future__ import division
from FilenameToCat import f2c
import Evaluation
from Tokenizer import get_list_tokens_nltk
from math import log
import Preprocessor
import time


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
    


##4)Create a) a dictionary with a category as the key and dictionary of words-occurences as values
          #b) a dictionary with a category as the key and the number of words in it as the value
CatWordDict={}
CatWordCountDict={}
#val = my_dict.get(key, mydefaultval)

##5)Loop through the training set, to get the entire text from  each file
##6) Parse the string to get individual words
for fileName in trainset:
    listWords = get_list_tokens_nltk(corpus,fileName)
    

##7) Check if category exists in dictionary, if not, create an empty dictionary,
    #and put word count as zero
    #and then insert words into the category's dictionary in both cases and update the word count
    cat = f2c(corpus,fileName)
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
    listWords = get_list_tokens_nltk(corpus,fileName)


    
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

    liResults.append((fileName,minCategory,f2c(corpus,fileName)))

###--------------------DEBUG STATEMENTS----------------------
#for t in liResults:
 #   if t[1]!=t[2]:
  #      print t
###--------------------DEBUG STATEMENTS----------------------
    
###--------------------DEBUG STATEMENTS----------------------

#12) Evaluating the classifier

if BinaryClassification:
    Evaluation.evaluation_binary(liResults)
else:
    Evaluation.evaluation_multi_class(liResults,CatNumDocs.keys())


print "The time taken by the trained classifier to assign labels"
print time.time() - start_time, "seconds"
