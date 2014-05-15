from __future__ import division
from nltk.corpus import movie_reviews as mr
from nltk.corpus import stopwords
import re
#from os import listdir
from os.path import isfile, join
from math import log
import time
start_time = time.time()

trainset=[]
testset=[]
trainToTestRatio=0.3
size=0
sw = stopwords.words('english')
swd={}

for w in sw:
    swd[w]=True
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
for cat in categoriesFilenameDict.keys():
    li = categoriesFilenameDict[cat]
    size=int(len(li)*trainToTestRatio)
    CatNumDocs[cat]=size
    trainset.extend(li[:size])
    testset.extend(li[size:])

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

for fileName in trainset:
    string = mr.raw(fileids=fileName)
    

##6) Parse the string to get individual words
#*************NON-GENERALIZED CODE***********************************
    #listWords = re.split(r'\s+',string)
    #listWords = [w[:w.find('/')] for w in listWords if w.find('/')!=-1  and w[w.find('/')+1:]=='jj']
    #listWords = [w.lower() for w in listWords if w.isalnum() and w not in sw]

    listWords = re.split(r'\W+',string)
    listWords = [w.lower() for w in listWords if w.isalnum() and w not in sw]
    #!!!!!!!!------Possible Improvement: Stemming--------------#
#********************************************************************    

##7) Check if category exists in dictionary, if not, create an empty dictionary,
    #and put word count as zero
    #and then insert words into the category's dictionary in both cases and update the word count
    cat = mr.categories(fileids=fileName)[0]
    if CatWordDict.get(cat, -1)==-1:
        CatWordDict[cat]={}
        CatWordCountDict[cat]=0

    CatWordCountDict[cat]+=len(listWords)            
    for w in listWords:
        if CatWordDict[cat].get(w, -21)==-21:
            CatWordDict[cat][w]=1
        else:
            CatWordDict[cat][w]+=1
        

##8) Get the vocabulary length
vocabLength=0            
for dic in CatWordDict.values():
     vocabLength+=len(dic)


print "The Classifier is trained and it took"
print time.time() - start_time, "seconds"
start_time = time.time()



####Congratulations! the Classifier is trained, now it is time to run the Multinomial Naive Bayes Classifier on the test dataset

liResults=[]
#9) Like in the training set,Loop through the test set, to get the entire text from  each file
for fileName in testset:
    minimumNegLogProb=1000000000
    minCategory=''
    string = mr.raw(fileids=fileName)

##10) Similar step, parse the string to get individual words
#*************NON-GENERALIZED CODE***********************************
    #listWords = re.split(r'\s+',string)
    #listWords = [w[:w.find('/')] for w in listWords if w.find('/')!=-1  and w[w.find('/')+1:]=='jj']
    #listWords = [w.lower() for w in listWords if w.isalnum() and w not in sw]

    listWords = re.split(r'\W+',string)
    listWords = [w.lower() for w in listWords if w.isalnum() and w not in sw]
    #!!!!!!!!------Possible Improvement: Stemming--------------#
#********************************************************************    
    
##11) Get the probability for each category,
    #can use any of the created dictionaries to wade through the categories
    for cat in  CatWordCountDict:
        #print cat , CatNumDocs[cat]/len(trainset)
        negLogProb=-log(CatNumDocs[cat]/len(trainset))
        wordDict = CatWordDict[cat]
        countCat = CatWordCountDict[cat]
        for w in listWords:
            countWordTrain=wordDict.get(w,0)
            ratio = (countWordTrain+1)/(countCat+vocabLength)
            negLogProb-=log(ratio)           
                         
        if minimumNegLogProb>negLogProb:
            minCategory=cat
            minimumNegLogProb=negLogProb

    liResults.append((fileName,minCategory,mr.categories(fileids=fileName)[0],minimumNegLogProb))

###--------------------DEBUG STATEMENTS----------------------
#for t in liResults:
 #   if t[1]!=t[2]:
  #      print t
###--------------------DEBUG STATEMENTS----------------------
    
###--------------------DEBUG STATEMENTS----------------------

#12) Calculate the precision, reacall and f-measure  
a=0
b=0
c=0
d=0
if len(CatWordCountDict)<3:
    cat = CatWordCountDict.keys()[0]
    for t in liResults:
        if cat==t[1]:
            if cat==t[2]:
                a+=1
            else:
                b+=1
        else:
            if cat==t[2]:
                c+=1
            else:
                d+=1
Precision = a/(a+b)
Recall = a/(a+c)

print "Precision =", Precision
print "Recall =", Recall
print "F-measure =", (2*Precision*Recall)/(Precision+Recall)

###--------------------DEBUG STATEMENTS----------------------
#print (a+b+c+d)==len(testset)
###--------------------DEBUG STATEMENTS----------------------

numErrors = sum(t[1]!=t[2] for t in liResults)
print "Fraction of Errors = ", numErrors/len(testset)



print "The time taken by the trained classifier to assign labels"
print time.time() - start_time, "seconds"
