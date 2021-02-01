import os
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,  f1_score
from utils import *



def classify_logreg(reviews_train_clean, target, reviews_test_clean, y_test):
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    ngram_vectorizer.fit(reviews_train_clean)
    X = ngram_vectorizer.transform(reviews_train_clean)
    X_test = ngram_vectorizer.transform(reviews_test_clean)

    # for c in [0.001, 0.01, 0.05, 0.25, 0.5, 1]:
    for c in [0.01]:
        lr = LogisticRegression(C=c)
        lr.fit(X, target)
        y_pred = lr.predict(X_test)
        print("Accuracy for C=%s: %s"
              % (c, accuracy_score(y_test, y_pred)),
              "F1_score for C=%s: %s" %(c, f1_score(y_test, y_pred)))
    return y_pred


def read_lexicon():
    with open('data/negative-words.txt', "r") as f_negative:
        negative_words = preprocess(f_negative.read()).split()
        negative_words = list(set(negative_words))
    with open('data/positive-words.txt', "r") as f_positive:
        positive_words = preprocess(f_positive.read()).split()
        positive_words = list(set(positive_words))

    return positive_words, negative_words


def load_corpus():
    negative_reviews = []
    positive_reviews = []
    X = {}
    for root, dir, file in os.walk('data/neg'):
        for filename in file:
            with open(f"{root}/{filename}", "r") as f:
                text = f.read()
                positive_reviews.append(preprocess(text))

    for root, dir, file in os.walk('data/pos'):
        for filename in file:
            with open(f"{root}/{filename}", "r") as f:
                text = f.read()
                negative_reviews.append(preprocess(text))

    return {"pos":positive_reviews, "neg":negative_reviews}

def shuffle(X, y):
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    return X, y

def split_train_test(X:dict, n_val = 400):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    sentences = []
    label = []
    for sentiment in ["pos", "neg"]:
        sentences.extend(X[sentiment])
    label = np.ones(len(X[sentiment])).tolist()
    label.extend(np.zeros(len(X[sentiment])).tolist())
    assert len(label) == len(sentences)
    dataset = list(zip(sentences, label))
    random.shuffle(dataset)
    sentences, label = zip(*dataset)
    X_train, X_test = sentences[n_val:], sentences[:n_val]
    y_train, y_test = label[n_val:], label[:n_val]
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    return X_train, X_test, y_train, y_test

def classify_by_count(X:str, positive, negative)-> int:
    count_pos = 0
    count_neg = 0
    X = X.split()

    for p in positive:
        count_pos += X.count(p)

    print("")
    for n in negative:
        count_neg += X.count(n)


    if count_neg > count_pos:
        return 0
    else:
        return 1



# Press the green button in the gutter to run the script.
def movie_review():

    positive_words, negative_words = read_lexicon()
    print("load corpus")
    X = load_corpus()
    X_train,  X_test, y_train, y_test = split_train_test(X, 400)

    print("classifier by Count")
    y_pred2 = []
    for X in X_test:
        pred = classify_by_count(X, positive_words, negative_words)
        y_pred2.append(pred)

    print("accuracy:", accuracy_score(y_test, y_pred2))
    print("F1_score:", f1_score(y_test, y_pred2))
    evaluate(y_pred2, y_test, "Counting ")

    print("classifier logreg")
    y_pred1 = classify_logreg(X_train, y_train, X_test, y_test)
    evaluate(y_pred1, y_test, "Logistic regression")

    print("Statistical correlation with mcNemar")
    p_value_mcNemar(y_test, y_pred1, y_pred2)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
