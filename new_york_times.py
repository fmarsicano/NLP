from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from utils import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from utils import p_value_mcNemar


def plot(x, y, title, x_label = 'x', y_label = 'y',):
    plt.figure()
    plt.xscale("log")
    plt.plot(x, y)
    plt.title(title)
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def load_corpus(size = 10e1):
    news = pd.read_csv('data/new_york_times.csv', sep = ',', nrows = size, encoding = "ISO-8859-1")
    topics = news.pop('majortopic')
    X = news.pop('summary') + ' ' + news.pop('title')
    df = X
    index = df.loc[pd.isna(df)].index
    X = X.fillna(' ')

    for i, x in enumerate(X):
        X[i] = preprocess(x)



    X_train, X_test, y_train, y_test = train_test_split(
        X, topics, test_size = 0.4, random_state = 42)

    # X_val, X_test, y_val, y_test = train_test_split(
    #     X_test, y_test, test_size = 0.5, random_state = 42)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def extract_features(X_train, vocab):
    ngram_vectorizer = CountVectorizer(binary=False, ngram_range=(1, 2), vocabulary=vocab)
    pipe = Pipeline([('count', ngram_vectorizer),('tfid', TfidfTransformer())])
    pipe.fit(X_train)
    return pipe


def classify_logreg(X_train, y_train, X_val,y_val, regularization):
    lr = LogisticRegression(penalty=regularization, max_iter=100000)

    if regularization == 'none':
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_val)
        print("Accuracy  %s"
              % (accuracy_score(y_val, y_pred)),
              "F1_score %s" %(f1_score(y_val, y_pred, average='macro')))
        return lr

    if regularization == 'l2':
        param_grid = {"C":[ 0.1, 1, 10, 1e2, 1e3, 1e4, 1e5]}
        grid = GridSearchCV(lr, param_grid, scoring='accuracy', cv =4)
        grid.fit(X_train, y_train)
        print("Best parameters were found for %s with a score of %s" %(grid.best_params_, grid.best_score_))
        return grid.best_estimator_

def get_vocabulary(corpus, vocab_size):
    word_list = []

    for k,  c in enumerate(corpus):
        word_list.extend(c.split())

    vocab = nltk.FreqDist(word_list)
    v1000 = list(vocab)[:vocab_size]

    return v1000


def new_york_times():
        print("load corpus")
        X_train, X_test, y_train,  y_test = load_corpus(1e7)
        sizes = [1e2, 1e3, 1e4, 1e5]
        f1_scores = []
        accuracy_scores = []

        # no regularization
        for dictionary_size in sizes:
            print("Size ", dictionary_size)
            vocab = get_vocabulary(X_train, int(dictionary_size))
            print("extract feature")
            extractor = extract_features(X_train, vocab)
            x_train = extractor.transform(X_train)
            x_test = extractor.transform(X_test)
            print("classify")
            clf = classify_logreg(x_train, y_train, x_test, y_test, regularization = 'none')
            f1_scores.append(f1_score(y_test, clf.predict(x_test), average= 'macro'))
            accuracy_scores.append(accuracy_score(y_test, clf.predict(x_test)))

        plot(sizes, f1_scores,"F1 scores with no regularization",  x_label= "dictionary size", y_label="f1 score")
        plot(sizes, accuracy_scores*100,"Accuracy scores with no regularization", x_label="dictionary_size", y_label="accuracy")
        print("")

        f1_scores = []
        accuracy_scores = []
        # l2 regularization
        for dictionary_size in sizes:
            vocab = get_vocabulary(X_train, int(dictionary_size))
            print("extract feature")
            extractor = extract_features(X_train, vocab)
            x_train = extractor.transform(X_train)
            x_test = extractor.transform(X_test)
            print("classify")
            clf = classify_logreg(x_train, y_train, x_test, y_test, regularization='l2')

            f1_scores.append(f1_score(y_test, clf.predict(x_test), average='macro'))
            accuracy_scores.append(accuracy_score(y_test, clf.predict(x_test)))
            print("scores", f1_scores, accuracy_scores)

        plot(sizes, f1_scores, "F1 scores with l2 regularization", x_label="dictionary size", y_label="f1 score",)
        plot(sizes, accuracy_scores,"Accuracy scores with l2 regularization", x_label="dictionary_size", y_label="accuracy")
        print("")