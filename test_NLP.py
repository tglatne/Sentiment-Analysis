# import required packages
import warnings
import os
import re
import csv

import keras.models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import train_NLP
from tensorflow.keras import layers
# from tensorflow.keras import losses

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings("ignore")
os.environ['PYTHONHASHSEED']=str(0)


def create_test_data():

    test_files = os.listdir("./data/aclImdb/test")
    data_columns = ['Review', 'Sentiment']
    test_df = pd.DataFrame(columns=data_columns)

    test_review_list = []
    test_senti = []

    for files in test_files:
        if files == 'pos':
            pos_files = os.listdir("./data/aclImdb/test/pos")
            for i in range(len(pos_files)):
                fo = open("./data/aclImdb/test/pos/" + pos_files[i], encoding="utf8")
                #             match = re.findall(pattern, pos_files[i])
                test_review_list.append(fo.readlines())
                test_senti.append(1)
        if files == 'neg':
            neg_files = os.listdir("./data/aclImdb/test/neg")
            for j in range(len(neg_files)):
                fo1 = open("./data/aclImdb/test/neg/" + neg_files[j], encoding="utf8")
                #             match = re.findall(pattern, neg_files[j])
                test_review_list.append(fo1.readlines())
                test_senti.append(0)

    test_review_list = [i for j in test_review_list for i in j]

    test_df['Review'] = test_review_list
    test_df['Sentiment'] = test_senti

    return test_df


def clean_data(reviews):

    pattern1 = '\"|\<[a-z]+ \/\>|\?\!+|[\(\)\+\/\d*]|(\?)'
    pattern2 = '[\.]+'
    pattern3 = '\-'
    pattern4 = ' \'|\''

    for i in range(len(reviews)):
        reviews[i] = re.sub(pattern1, '', reviews[i])
        reviews[i] = re.sub(pattern2, '.', reviews[i])
        reviews[i] = re.sub(pattern3, ' ', reviews[i])
        reviews[i] = re.sub(pattern4, '', reviews[i])
        reviews[i] = reviews[i].lower()
    return reviews

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__": 
	# 1. Load your saved model

    model = keras.models.load_model("./models/Group31_NLP_model/saved_model.pb")

	# 2. Load your testing data
    num_words = 10000
    oov_token = '<UNK>'
    pad_type = 'post'
    trunc_type = 'post'

    test_data_df = create_test_data()

    reviews = test_data_df['Review']
    test_data_df['Review'] = clean_data(reviews)

    train_data_review = train_NLP.train_data_review_list
    test_data_df = test_data_df.sample(frac=1).reset_index(drop=True)

    test_data_review_list = test_data_df['Review'].tolist()
    test_data_sentiment_list = test_data_df['Sentiment'].tolist()

    test_tokenizer = Tokenizer(num_words= num_words, oov_token= oov_token)
    test_tokenizer.fit_on_texts(train_data_review)

    test_word_index = test_tokenizer.word_index

    test_sequences = test_tokenizer.tests_to_sequences(test_data_review_list)
    test_maxlen = max([len(x) for x in test_sequences])

    test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=test_maxlen)

    test_y_label = np.array(test_data_sentiment_list)

	# 3. Run prediction on the test data and print the test accuracy

    loss, accuracy = model.evaluate(test_padded, test_y_label, batch_size=128)

    print(accuracy)