import nltk
import os
import typing
import numpy as np
import random
import string
import pandas as pd
import seaborn as sn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
# Example of calculating the mcnemar test
from statsmodels.stats.contingency_tables import mcnemar
nltk.download('wordnet')


def get_stemmed_text(sentence):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    stemmed = ' '.join([stemmer.stem(word) for word in sentence.split()])
    return stemmed

def get_lemmatized_text(sentence):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in sentence.split()])

def evaluate(y_pred, y_test, title):
    m = confusion_matrix(y_pred, y_test)
    mcc = matthews_corrcoef(y_pred, y_test)
    df_cm = pd.DataFrame(m, index = ['Predicted positive', 'Predicted negative'], columns = ['Actual positive', 'Actual negative'])
    plt.figure(figsize=(14, 7))
    # plt.title(title)
    sn.set(font_scale=3.0)  # Adjust to fit
    sn.heatmap(df_cm, annot=True,  fmt='g')
    plt.show()

def preprocess(text:str)->str:
    text = text.replace("\n", " ")
    text = text.replace("\\'", "'")
    text = text.lower()
    text = get_lemmatized_text(text)
    text = get_stemmed_text(text)
    text = remove_stop_words(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def remove_stop_words(review):
    english_stop_words = stopwords.words('english')
    removed_stop_words = []
    removed_stop_words = \
        ' '.join([word for word in review.split()
                  if word not in english_stop_words])
    return removed_stop_words

def p_value_mcNemar(y_test, y_pred1, y_pred2):
    #     contingency table
    ct = np.zeros((2, 2))
    for k, y in enumerate(y_test):
        if  y == y_pred1[k] and y == y_pred2[k]:
            ct[0, 0]+=1
        elif y != y_pred1[k] and y == y_pred2[k]:
            ct[1, 0] += 1
        elif y != y_pred1[k] and y != y_pred2[k]:
            ct[1, 1] +=1
        elif y == y_pred1[k] and y != y_pred2[k]:
            ct[0, 1]+=1

    print(ct)
    pd_ct = pd.DataFrame(ct, columns=['C2 correct', 'C2 incorrect'], index=['C1 correct', 'C1 incorrect'])
    plt.figure(figsize=(14, 7))
    plt.title("Contingency table")
    sn.set(font_scale=3.0)  # Adjust to fit
    sn.heatmap(pd_ct, annot=True, fmt='g')
    plt.show()
    print(pd_ct)
    result = mcnemar(table=ct, exact=False, correction=True)
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
