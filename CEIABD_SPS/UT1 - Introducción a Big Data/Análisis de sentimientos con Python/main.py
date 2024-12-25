from textblob import TextBlob
import pandas as pd

# Reemplaza 'ruta_del_archivo.csv' con la ruta de tu archivo CSV
df = pd.read_csv('CEIABD_SPS\\UT1 - Introducción a Big Data\\Análisis de sentimientos con Python\\data.csv')

numberOfSentences = int(input("Introduzca el número de sentencias a analizar: "))
for index, row in df.iterrows():
    text = row['Sentence'] 
    sentiment = row['Sentiment'] 
    print(f"SENTENCE: \"{text}\", || SENTIMENT: {sentiment}")
    print("Polarity of Text 1 is",TextBlob(text).sentiment.polarity)
    print("Subjectivity of Text 1 is", TextBlob(text).sentiment.subjectivity)
    if index >= numberOfSentences:
        break