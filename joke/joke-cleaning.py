import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from string import punctuation

# nltk.download()
# nltk.download('punkt')
# if you have not downloaded the nltk corpus, then uncomment the lines above

def parseJoke(filename):
    data = pd.read_csv(filename)
    return data

def CreateMyStopWords ():
    stopword = stopwords.words("english")
    stopword.remove(u't')
    stopword.remove(u's')
    stopword.append(u"'s")
    stopword.append(u"'t")
    stopword.append(u"n't")
    stopword.append(u"'d")
    stopword.append(u"'re")
    stopword.append(u"cannot")
    stopword.append(u"'ll")
    stopword.append(u"'ve")
    stopword.append(u"'m")
    stopword.append(u"q")
    stopword.append(u"could")
    stopword.append(u"would")
    return stopword
    
def is_valid_hyphen_word(str):
    flag = False
    
    if str[0].isalpha() and str[len(str) - 1].isalpha():
        for chr in str:
            if chr.isalpha():
                flag = False
            elif chr == "-":
                if flag:
                    return False
                else:
                    flag = True
            else:
                return False
        return True
    return False

def DataCleaningForKaggleSA(data):
    stopword = CreateMyStopWords()
    porterStemmer = PorterStemmer()
    
    for i in range(len(data)):
        row = data.iloc[i]
        sentence = row["Joke"].replace("’", "'").lower()
        for chr in sentence:
            if (ord(chr) >= 128):
                sentence = sentence.replace(chr, '')
                
        words = word_tokenize(sentence)
        cleanData = []
        
        for w in words:
            if w not in stopword:
                if all(chr not in punctuation for chr in w) or is_valid_hyphen_word(w):
                    cleanData.append(porterStemmer.stem(w))
            
        cleanSentence = ' '.join(cleanData)
        data.set_value(i, "Joke", cleanSentence)
        
    return data

data = parseJoke("jokes.csv")

DataCleaningForKaggleSA(data).to_csv("cleanedJokes.csv")
