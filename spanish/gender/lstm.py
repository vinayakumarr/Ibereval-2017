import numpy as np
import pandas as pd

from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM
from sklearn import preprocessing
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
np.random.seed(0)
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

if __name__ == "__main__":

    train_df = pd.read_csv('data/training_tweets_es.txt', sep='\t',header=0)
    classlabels = pd.read_csv('data/training_truth_es.txt', header=0)
   
    test_df = pd.read_csv('data/testing_tweets_es.txt', sep='\t',header=0)
    testclasslabels = pd.read_csv('data/testing_truth_es.txt', header=0)
  
    raw_docs_train = train_df['phrase'].values
    classlabels1 = classlabels['label'].values
    raw_docs_test = train_df['phrase'].values
    testclasslabels1 = classlabels['label'].values

            
    stop_words = set(stopwords.words('dutch'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stemmer = SnowballStemmer('dutch')

    print "pre-processing train docs..."
    processed_docs_train = []
    for doc in raw_docs_train:
       doc = doc.decode("utf8")
       tokens = word_tokenize(doc)
       filtered = [word for word in tokens if word not in stop_words]
       stemmed = [stemmer.stem(word) for word in filtered]
       processed_docs_train.append(stemmed)
   
    print "pre-processing test docs..."
    processed_docs_test = []
    for doc in raw_docs_test:
       doc = doc.decode("utf8")
       tokens = word_tokenize(doc)
       filtered = [word for word in tokens if word not in stop_words]
       stemmed = [stemmer.stem(word) for word in filtered]
       processed_docs_test.append(stemmed)

    processed_docs_all = np.concatenate((processed_docs_train, processed_docs_test), axis=0)


    dictionary = corpora.Dictionary(processed_docs_all)
    dictionary_size = len(dictionary.keys())
    print "dictionary size: ", dictionary_size 
    
    print "converting to token ids..."
    word_id_train, word_id_len = [], []
    for doc in processed_docs_train:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_train.append(word_ids)
        word_id_len.append(len(word_ids))
    
    word_id_test, word_ids = [], []
    for doc in processed_docs_test:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_test.append(word_ids)
        word_id_len.append(len(word_ids))

    seq_len = np.round((np.mean(word_id_len) + 2*np.std(word_id_len))).astype(int)

    #pad sequences
    word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
    word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)
       
    num_labels = 1
    
    y_train_enc = np.array(classlabels1)
    y_test_enc = np.array(testclasslabels1)
    #LSTM
    print "fitting LSTM ..."
    model = Sequential()
    model.add(Embedding(dictionary_size, 256, dropout=0.2))
    model.add(LSTM(256, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(num_labels))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpointer = callbacks.ModelCheckpoint(filepath="logs/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
    csv_logger = CSVLogger('logs/training_set_iranalysis1.csv',separator=',', append=False)

    model.fit(word_id_train, y_train_enc, nb_epoch=1000, batch_size=32, validation_data=(word_id_test, y_test_enc), verbose=1, callbacks=[checkpointer,csv_logger])
    model.save("logs/lstm_model.hdf5")
    

