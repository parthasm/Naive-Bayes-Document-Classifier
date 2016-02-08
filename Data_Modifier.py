from nltk.corpus import reuters
import re

def write_into_new_file(file_name):
    #stopwords_english = set(stopwords.words('english'))
    string = reuters.raw(fileids=file_name)
    list_words = re.split(r'\W+',string)
    new_file_path = new_path+file_name
    file_wr = open(new_file_path, "w")
    for w in list_words:
        if w.isalpha() and len(w)>1 and w.lower() not in stopwords_english:
            file_wr.write(w.lower()+"\n")
    file_wr.close()
if __name__ == '__main__':
    orig_path = "D:/Zmisc/Github/NLP/Naive-Bayes-Document-Classifier-master/trial/reuters/"
    new_path = "D:/Zmisc/Github/NLP/Naive-Bayes-Document-Classifier-master/trial/reuters_modified/"
    
    stopwords_english = set()
    for line in open(orig_path+"stopwords"):
        stopwords_english.add(line.strip())
    
    #print stopwords_english
    
    for file_name in reuters.fileids():
        write_into_new_file(file_name)
        
            
