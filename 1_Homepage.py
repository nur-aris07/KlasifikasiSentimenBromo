import streamlit as st
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

st.set_page_config(
    page_title="Analisis Sentimen Bromo",
    page_icon=":sun_with_face:",
    layout = "wide"
)

st.title("Klasifikasi Sentimen Komentar Bromo pada Twitter dengan K-Nearest Neighbor")

st.header("Dataset")
df = pd.read_csv('sentimen-bromo-twitter.csv',encoding= 'unicode_escape',sep=';')
data = df[['Text','Label']]
st.write(data)
data_teks = data.loc[:,"Text"]

st.header("Preprocessing")

col1, col2= st.columns([0.5,0.5])
with col1:
    st.subheader("1) Cleaning")
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
    data_teks = np.vectorize(remove_pattern)(data_teks, '@[\w]*')
    data_teks = np.vectorize(remove_pattern)(data_teks, r'http\S+')
    data_teks = np.vectorize(hapus_karakter)(data_teks)
    st.write(data_teks)
with col2:
    st.subheader("2) Case Folding")
    def clean_lower(lwr):
        lwr = lwr.lower()
        return lwr
    data_teks = np.vectorize(clean_lower)(data_teks)
    st.write(data_teks)

with col1:
    st.subheader("3) Stop Removal")
    def stopRemoval(data):
        factory = StopWordRemoverFactory()
        stopwords = factory.get_stop_words()
        stopwords = ArrayDictionary(stopwords)
        stopword = StopWordRemover(stopwords)
        for i in range(len(data)):
            data[i] = stopword.remove(data[i])
        return data
    data_teks = stopRemoval(data_teks)
    st.write(data_teks)
with col2:
    st.subheader("4) Stemming")
    def stemming(data):
        fact = StemmerFactory()
        stemmer = fact.create_stemmer()
        for i in range(len(data)):
            data[i] = stemmer.stem(data[i])
        return data
    data_teks = stemming(data_teks)
    st.write(data_teks)

st.subheader("5) Tokenisasi")
nltk.download('punkt')
count_vector = CountVectorizer(encoding='latin-1', analyzer='word')
data_array = count_vector.fit_transform(data_teks)
cekout = count_vector.fit_transform(data_teks[:2])
st.markdown(cekout)

x = data_array
y = data.loc[:,'Label'].values
x_train = x[:70]
x_test = x[-30:]
y_train = y[:70]
y_test = y[-30:]

st.header("Model Klasifikasi")

classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(x_train,y_train)
pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, pred)
akurasi = (cm[0][0]+cm[1][1])/len(y_test)*100

def getSentimen(data):
    positif = 0
    negatif = 0
    for sentimen in data:
        if sentimen=="POSITIF":
            positif += 1
        elif sentimen=="NEGATIF":
            negatif += 1
    return [positif,negatif]

st.markdown(f"Nilai akurasi : {akurasi}%",unsafe_allow_html=True)
st.subheader("Hasil Prediksi")

col3, col4= st.columns([0.5,0.5])
with col3:
    tabeldf = pd.DataFrame()
    tabeldf["prediksi"] = pred
    tabeldf["aktual"] = y_test
    st.write(tabeldf)
with col4:
    chartdf = pd.DataFrame([getSentimen(pred),getSentimen(y_test)],columns=["Positif","Negatif"])
    st.bar_chart(data=chartdf,x=None,y=None)