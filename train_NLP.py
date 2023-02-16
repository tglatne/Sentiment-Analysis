# import required packages
import warnings
import os
import re
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras import layers
# from tensorflow.keras import losses

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

global train_data_review_list
# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
warnings.filterwarnings("ignore")
os.environ['PYTHONHASHSEED']=str(0)

def create_train_data():
    """The function creates the required data to train the model"""

    train_files = os.listdir("./data/aclImdb/train")
    data_columns = ['Review', 'Sentiment']
    train_df = pd.DataFrame(columns=data_columns)

    reviews_list = []
    sentiment_list = []

    for files in train_files:
        if files == 'pos':
            pos_files = os.listdir("./data/aclImdb/train/pos")
            for i in range(len(pos_files)):
                fo = open("./data/aclImdb/train/pos/" + pos_files[i], encoding="utf8")
                #             match = re.findall(pattern, pos_files[i])
                reviews_list.append(fo.readlines())
                sentiment_list.append(1)
        if files == 'neg':
            neg_files = os.listdir("./data/aclImdb/train/neg")
            for j in range(len(neg_files)):
                fo1 = open("./data/aclImdb/train/neg/" + neg_files[j], encoding="utf8")
                #             match = re.findall(pattern, neg_files[j])
                reviews_list.append(fo1.readlines())
                sentiment_list.append(0)

    reviews_list = [i for j in reviews_list for i in j]
    # sentiment_list = [x for y in sentiment_list for x in y]

    train_df['Review'] = reviews_list
    train_df['Sentiment'] = sentiment_list

    # Final Train data for NLP
    train_df.to_csv("./NLP_train_data.csv", index=False, header=True)

    return train_df


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


if __name__ == "__main__":
	# 1. load your training data

    num_words = 10000
    oov_token = '<UNK>'
    pad_type = 'post'
    trunc_type = 'post'

    train_data_df = create_train_data()

    # Reading the created train data
    # train_data_df = pd.read_csv("./NLP_train_data.csv")
    reviews = train_data_df['Review']
    train_data_df['Review'] = clean_data(reviews)
    # print(train_data_df)

    # Shuffling the rows of the train data
    train_data_df = train_data_df.sample(frac=1).reset_index(drop=True)
    # print(train_data_df.head())
    train_data_review_list = train_data_df['Review'].tolist()
    train_data_sentiment_list = train_data_df['Sentiment'].tolist()

    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(train_data_review_list)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_data_review_list)
    maxlen = max([len(x) for x in train_sequences])

    train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

    y_label = np.array(train_data_sentiment_list)

    X_train, X_val, y_train, y_val = train_test_split(train_padded, y_label, test_size=0.3, random_state=121)

    model = tf.keras.Sequential([
        layers.Embedding(len(word_index) + 1, 16),
        layers.Conv1D(filters=32, kernel_size=3, padding='valid', activation='relu', strides=1),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(250, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')])

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model.fit(tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train), validation_data=(tf.convert_to_tensor(X_val), tf.convert_to_tensor(y_val)), epochs=15, batch_size=128)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=128)

# 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy

	# 3. Save your model
    # model.save("./models/Group31_NLP_model", save_format="h5")