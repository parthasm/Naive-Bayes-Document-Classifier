from nltk.corpus import reuters
from nltk.corpus import movie_reviews as mr

def reuters_f2c(fileName):
    return reuters.categories(fileids = fileName)[0]

def movie_Reviews_f2c(fileName):
    return mr.categories(fileids = fileName)[0]

