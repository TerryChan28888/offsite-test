
# author: Chan Ho Yin
# student_id: 20311091
import numpy as np
from scipy import sparse
import nltk
import pandas as pd
from sklearn import datasets, svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from statistics import mean
from numpy import linspace, zeros
import jieba
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

jieba.set_dictionary('data/dict.txt.big')
def tokenize(text):
    '''
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
    e.g. 
    Input: "多獲一次機會的紅軍不敢再有差池"
    Output: ['多獲', '一次', '機會', '的', '紅軍', '不敢', '再有', '差池']
    '''
    tokens = []
    # YOUR CODE HERE
    for word in jieba.cut(text):
        # word = word.lower()
        #try to not use stop words
        # if word not in stop_words and not word.isnumeric():
        tokens.append(word)

    return tokens

def get_bagofwords(data, vocab_dict):
    '''
    :param data: a list of words, type: list
    :param vocab_dict: a dict from words to indices, type: dict
    return a word (sparse) matrix, type: scipy.sparse.csr_matrix
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html
    '''
    #data_matrix = None
    # YOUR CODE HERE
    data_matrix = sparse.lil_matrix( (len(data), len(vocab_dict)) )
    for i, doc in enumerate(data):
        for word in doc:
            # dict.get(key, -1)
            # if the word in the vocab_dic, return the value
            # else return -1
            word_idx = vocab_dict.get(word, -1)
            if word_idx != -1:
                data_matrix[i, word_idx] += 1
    # data_matrix = data_matrix.tocsr() #to speed up when computation
    return data_matrix

def read_data(file_name, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    df['words'] = df['text'].apply(tokenize)

    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = dict(zip(vocab, range(len(vocab))))

    data_matrix = get_bagofwords(df['words'], vocab_dict)
    # print(data_matrix)
    return df['id'], df['tags'], data_matrix, vocab

def read_test_data(file_name, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    df['words'] = df['text'].apply(tokenize)

    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = dict(zip(vocab, range(len(vocab))))

    data_matrix = get_bagofwords(df['words'], vocab_dict)
    # print(data_matrix)
    return df['id'], data_matrix


if __name__ == '__main__':




    # seg_list = jieba.cut("英格蘭足總盃第三圈今晨重賽，貴為英超勁旅的利物浦上場被乙組仔埃克斯特尷尬逼和，多獲一次機會的紅軍不敢再有差池。", cut_all=False)
    # print(tokenize("多獲一次機會的紅軍不敢再有差池"))
    # print("default mode: "+"/".join(seg_list))

    train_id_list, train_data_label, train_data_matrix, vocab = \
        read_data("offsite-test-material/offsite-tagging-training-set (1).csv")

    print("Training Set Size:", len(train_id_list))

    test_id_list, test_data_matrix, = read_test_data("offsite-test-material/offsite-tagging-test-set (1).csv", vocab)
    print("Test Set Size:", len(test_id_list))


    # currently, the best performance
    # clf = SVC(kernel='linear', C=0.0023)
    # clf.fit(train_data_matrix, train_data_label)  # there is clf.predict()
    # y_hat = clf.predict(test_data_matrix)

    # this is for training error
    # linear_x_axis_1 = linspace(0.5, 2, num=5)
    # linear_y_axis_1 = zeros(10)
    # print("cross validation")
    # for i in range(len(linear_x_axis_1)):
    #     clf = SVC(kernel='linear', C=linear_x_axis_1[i])
    #     clf.fit(train_data_matrix, train_data_label)
    #     scores = cross_val_score(clf, train_data_matrix, train_data_label, cv=5)
    #     print("linear kernel, C =", end=" ")
    #     print(linear_x_axis_1[i], end=" ")
    #     print(", training accuracy =", end=" ")
    #     print(mean(scores))

    # linear kernel, C = 0.0033 , training accuracy = 0.9897264432920851 (0.0023-0.0033)

    # linear kernel, C = 0.01 , training accuracy = 0.9897264432920851
    # linear kernel, C = 0.021111111111111112 , training accuracy = 0.9889562225218643
    # linear kernel, C = 0.03222222222222222 , training accuracy = 0.9879282723411542 (0.01-0.11) (0.5-2)only give same

    # trial for logistic regression
    clf = LogisticRegression(random_state=0, solver='newton-cg', multi_class='multinomial').fit(train_data_matrix,train_data_label)

    # scores = cross_val_score(clf, train_data_matrix, train_data_label, cv=5)
    # print("logistic regression, ")
    # print(", training accuracy =", end=" ")
    # print(mean(scores))

    # with n_jobs, 10:49-10:53, while w/o it, 10:54-10:58

    # sag : acc is around 0.987
    # saga : acc is around 0.971
    # liblinear : acc is around (why one-vs-rest is not used by machine?)
    # newton-cg : acc is around 0.991  (, training accuracy = 0.9907530726231454) !!!!!!!!!
    # lbfgs : acc is around 0.990  (, training accuracy = 0.9902382746500675) !!!!!!!!!

    # with traditional chinese dict,
    # newton-cg : (, training accuracy = 0.9917793711055457)
    # saga : acc is around 0.974
    # lbfgs : (, training accuracy = 0.991266552288296)

    y_hat = clf.predict(test_data_matrix)

    sub_df = pd.DataFrame()
    sub_df["id"] = test_id_list
    sub_df["pred"] = y_hat
    sub_df.to_csv("submission.csv", index=False)
