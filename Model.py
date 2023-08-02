import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('sentimen-bromo-twitter.csv',encoding= 'unicode_escape',sep=';')
data = df[['Text','Label']]
data_teks = data.loc[:,"Text"]

def remove_pattern(tweet, pattern):
    r = re.findall(pattern, tweet)
    for i in r:
        tweet = re.sub(i, '', tweet)
    return tweet
def hapus_karakter(tweet):
    tandabaca = [".",",","%","?","!",":",";","&","%", "_", "#"]
    for td in tandabaca:
        tweet = np.char.replace(tweet, td, "")
    return tweet

def clean_lower(lwr):
    lwr = lwr.lower()
    return lwr

def stopRemoval(data):
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    stopwords = ArrayDictionary(stopwords)
    stopword = StopWordRemover(stopwords)
    for i in range(len(data)):
        data[i] = stopword.remove(data[i])
    return data

def stemming(data):
    fact = StemmerFactory()
    stemmer = fact.create_stemmer()
    for i in range(len(data)):
        data[i] = stemmer.stem(data[i])
    return data

data_teks = np.vectorize(remove_pattern)(data_teks, '@[\w]*')
data_teks = np.vectorize(remove_pattern)(data_teks, r'http\S+')
data_teks = np.vectorize(hapus_karakter)(data_teks)
data_teks = np.vectorize(clean_lower)(data_teks)
data_teks = stopRemoval(data_teks)
data_teks = stemming(data_teks)
nltk.download('punkt')
count_vector = CountVectorizer(encoding='latin-1', analyzer='word')
data_array = count_vector.fit_transform(data_teks)

x = data_array
y = data.loc[:,'Label'].values
x_train = x[:70]
x_test = x[-30:]
y_train = y[:70]
y_test = y[-30:]

classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(x_train,y_train)
pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, pred)
akurasi = (cm[0][0]+cm[1][1])/len(y_test)*100

print(akurasi)