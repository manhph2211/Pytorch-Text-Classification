
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


def saveVocab(path,data_texts_1,model):
    vocab,EMBEDDING_SIZE = getDic()
    with open(path,'w'):
        pass


# def encodingLabels(labels,class_num):
#   onehotLabels=[]
#   for label in labels:
#     l=[0]*class_num
#     l[label]=1
#     onehotLabels.append(l)
#   return onehotLabels

# labels=encodingLabels(labels,class_num)


def encodingData(data_texts_2,vocab):
  data=[]
  for text in data_texts_2:
    text_to_vec=[]
    for word in text.split():
      text_to_vec.append(vocab[word])
    data.append(text_to_vec)
  return data
