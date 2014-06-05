from nltk.corpus import stopwords
from nltk.corpus import reuters
from nltk.corpus import movie_reviews as mr
import re

sw = stopwords.words('english')
swd={}

for w in sw:
    swd[w]=True


def get_list_tokens_nltk_reuters(fileName):
    string = reuters.raw(fileids=fileName)
    listWords = re.split(r'\W+',string)
    return [w.lower() for w in listWords if w.isalpha() and len(w)>1 and not swd.get(w.lower(),False)]

#!!!!!!!!------Possible Improvement: Stemming--------------#

def get_list_tokens_nltk_mr(fileName):
    string = mr.raw(fileids=fileName)
    listWords = re.split(r'\W+',string)
    return [w.lower() for w in listWords if w.isalpha() and len(w)>2 and not swd.get(w.lower(),False)]
#*************NON-GENERALIZED CODE***********************************
    #listWords = re.split(r'\s+',string)
    #listWords = [w[:w.find('/')] for w in listWords if w.find('/')!=-1  and w[w.find('/')+1:]=='jj']
    #listWords = [w.lower() for w in listWords if w.isalnum() and w not in sw]

    
    #!!!!!!!!------Possible Improvement: Stemming--------------#
#********************************************************************    
