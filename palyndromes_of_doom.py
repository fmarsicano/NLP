import numpy as np
from sklearn.preprocessing import OneHotEncoder
import nltk


def check_palyndrome(string):
    return  string[::-1] == string

def fill_matrix(string):
    N = len(string)
    arr = np.ones((N, N))*(0)
    for i in range(N):
        for j in range(i+1, N):
            if check_palyndrome(string[i:(j+1)]):
                arr[i, j] = j-i + 1

    return arr

def find_longest(arr, string):
    N = len(string)
    string_values = np.zeros(N)
    for i in range(1, len(string)):
        print(arr[:i, i])
        string_values[i] = np.max(string_values[:i] + arr[:i, i])
    return string_values

def score():
    vocab = nltk.FreqDist(word_list)

if __name__ == "__main__":
    string = "abaca"
    arr = fill_matrix(string)
    print(arr)
    longest = find_longest(arr, string)

    print(longest)

