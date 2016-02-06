from nltk.corpus import stopwords
if __name__ == "__main__":
    stopwords_english = list(set(stopwords.words('english')))
    file_write_obj_stopwords = open("stopwords.txt","w")
    for w in stopwords_english:
        file_write_obj_stopwords.write(w+"\n")
    file_write_obj_stopwords.close()
    
