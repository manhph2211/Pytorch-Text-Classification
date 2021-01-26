
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn import preprocessing


def readCSV(path):
	data_df = pd.read_csv(path)
	le = preprocessing.LabelEncoder()
	data_texts = data_df['Questions'].to_list()
	labels = le.fit_transform(data_df['Category0'])

	return data_texts,labels


def readModel(path):
	model = KeyedVectors.load_word2vec_format(path, binary=True)
	return model


def saveVocab(path,data_texts_1,model):
    vocab,EMBEDDING_SIZE = getDic()
    with open(path,'w'):
        pass
