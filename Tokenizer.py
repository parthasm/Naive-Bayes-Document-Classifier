from nltk.corpus import stopwords
import re

sw = set(stopwords.words('english'))

def get_list_tokens_nltk(corpus, fileName):
    string=''
    if corpus=='mr':
        from nltk.corpus import movie_reviews
        string = movie_reviews.raw(fileids=fileName)
    else:
        from nltk.corpus import reuters
        string = reuters.raw(fileids=fileName)
    listWords = re.split(r'\W+',string)
    return [w.lower() for w in listWords if w.isalpha() and len(w)>1 and w.lower() not in sw]

#!!!!!!!!------Possible Improvement: Stemming--------------#

#*************NON-GENERALIZED CODE***********************************
    #listWords = re.split(r'\s+',string)
    #listWords = [w[:w.find('/')] for w in listWords if w.find('/')!=-1  and w[w.find('/')+1:]=='jj']
    #listWords = [w.lower() for w in listWords if w.isalnum() and w not in sw]

    
    #!!!!!!!!------Possible Improvement: Stemming--------------#
#********************************************************************    
