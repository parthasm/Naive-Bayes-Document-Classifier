from __future__ import division
#from nltk.corpus import brown
from nltk.corpus import stopwords
import re
#from os import listdir
from os.path import isfile, join
from math import log

trainset=[]
testset=[]
trainToTestRatio=0.3
size=0
sw = stopwords.words('english')

##1)alternate way, to be implemented, get the categories and the corresponding
##list of files from the user,
#Here we are getting it using nltk
categoriesFilenameDict={}

#*************NON-GENERALIZED TEXT***********************************


#for category in brown.categories()[:-1]:
 #   categoriesFilenameDict[category]=brown.fileids(categories=category)

 
categoriesFilenameDict['China']=[]
categoriesFilenameDict['Japan']=[]
for i in range(1,16):
    if i==4 or i>11:
        categoriesFilenameDict['Japan'].append(str(i))
    else:
        categoriesFilenameDict['China'].append(str(i))
#********************************************************************
    
#This categoriesFileidDict should be obtained from the user.
#*****Helpfully, in the brown corpus, fileids are same as file names   
    
##2)Forming a) the reverse dictionary - file to category
          ##b)Prepare the CatNumDocs dictionary, where the number of documents in the training set for each
             ##category are stored
    ##also forming the training set and test set
FilenameCategoriesDict={}
CatNumDocs={}
for cat in categoriesFilenameDict.keys():
    li = categoriesFilenameDict[cat]
    size=int(len(li)*trainToTestRatio)
    CatNumDocs[cat]=size
    trainset.extend(li[:size])
    testset.extend(li[size:])
    for f in li:
        FilenameCategoriesDict[f]=cat

###--------------------DEBUG STATEMENTS----------------------
#for f in trainset:
 #   print f , FilenameCategoriesDict[f] 

#print "Freedom\n"

#for f in testset:
 #   print f , FilenameCategoriesDict[f]    
###--------------------DEBUG STATEMENTS----------------------
    

##3)Reach folder taking user input
folderPath = input("Enter the path for the folder containing the text files of your corpus,"\
             "with escape characters and within quotes: ")
#onlyfileNames = [ f for f in listdir(folderPath) if isfile(join(folderPath,f)) ]
#fileHandle = open(join(folderPath,trainset[0]))
#string = fileHandle.read()
#print string
#fileHandle.close()

##4)Create a) a dictionary with a category as the key and dictionary of words-occurences as values
          #b) a dictionary with a category as the key and the number of words in it as the value
CatWordDict={}
CatWordCountDict={}
#val = my_dict.get(key, mydefaultval)

##5)Loop through the training set, to get the entire text from  each file

for fileName in trainset:
    fileHandle = open(join(folderPath,fileName))
    string = fileHandle.read()
    fileHandle.close()

##6) Parse the string to get individual words
#*************NON-GENERALIZED TEXT***********************************
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
    cat = FilenameCategoriesDict[fileName]
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


####Congratulations! the Classifier is trained, now it is time to run the Multinomial Naive Bayes Classifier on the test dataset

liResults=[]
#9) Like in the training set,Loop through the test set, to get the entire text from  each file
for fileName in testset:
    minimumNegLogProb=1000000000
    minCategory=''
    fileHandle = open(join(folderPath,fileName))
    string = fileHandle.read()
    fileHandle.close()

##10) Similar step, parse the string to get individual words
#*************NON-GENERALIZED TEXT***********************************
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

    liResults.append((fileName,minCategory,FilenameCategoriesDict[fileName]))

###--------------------DEBUG STATEMENTS----------------------
#for t in liResults:
 #   print t    
###--------------------DEBUG STATEMENTS----------------------
    
#12) Create a dictionary with category as the key and a list of 4 numbers as the value
 #These values are a) Number of docs in the category identified correctly a
                  #b) Number of docs identified incorrectly as in the category b
                  #c) Number of docs identified incorrectly as not in the category c
                  #d) Number of docs identified correctly as not in the category d
CatResultsDict = {}
for cat in CatWordCountDict:
    CatResultsDict[cat]=[0,0,0,0]
    for t in liResults:
        if cat==t[1]:
            if cat==t[2]:
                CatResultsDict[cat][0]+=1
            else:
                CatResultsDict[cat][1]+=1
        else:
            if cat==t[2]:
                CatResultsDict[cat][2]+=1
            else:
                CatResultsDict[cat][3]+=1

totPrec=0
totRec=0
A=0
B=0
C=0
D=0
for cat in CatResultsDict:
    a = CatResultsDict[cat][0]
    b = CatResultsDict[cat][1]
    c = CatResultsDict[cat][2]
    d = CatResultsDict[cat][3]
    totPrec+=a/(a+b)##Precision for this category
    totRec+=a/(a+c)##Recall for this category
    A+=a
    B+=b
    C+=c
    D+=d
###--------------------DEBUG STATEMENTS----------------------
    #print cat, a
    #print cat, b
    #print cat, c
    #print cat, d
    #print (a+b+c+d)==len(testset)
###--------------------DEBUG STATEMENTS----------------------
MacroPrec = totPrec/len(CatResultsDict)
MacroRec = totRec/len(CatResultsDict)
MacroF = (2*MacroPrec*MacroRec)/(MacroPrec+MacroRec)

MicroPrec = A/(A+B)
MicroRec = A/(A+C)
MicroF = (2*MicroPrec*MicroRec)/(MicroPrec+MicroRec)

print "Macro Precision=",MacroPrec
print "Macro Recall=",MacroRec
print "Macro F-measure=",MacroF

print "Micro Precision=",MicroPrec
print "Micro Recall=",MicroRec
print "Micro F-measure=",MicroF

numErrors = sum(t[1]!=t[2] for t in liResults)
print numErrors/len(testset)