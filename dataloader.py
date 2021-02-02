import numpy as np
from preprocessing import encodingData,getRidOfXXX,getMaxLen,getDic,padding,encodingData
from utils import readCSV, readModel, saveVocab
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset,DataLoader

def loadData(csv_path='./Question_Classification_Dataset.csv',model_path='./GoogleNews-vectors-negative300.bin'):
	
	data_texts,labels=readCSV(csv_path)
	print("1")
	model=readModel(model_path)
	print("2")
	data_texts_1=getRidOfXXX(data_texts,model)
	print("3")
	max_len=getMaxLen(data_texts_1)
	print("4")
	vocab,EMBEDDING_SIZE=getDic(data_texts_1,model)
	print("5")
	data_texts_2=padding(data_texts_1,max_len)
	print("6")
	data=torch.tensor(encodingData(data_texts_2,vocab))
	print("7")
	targets=torch.tensor(labels,dtype=torch.long)
	print("8")
	return data,targets,EMBEDDING_SIZE


def splitData(data,targets):

	X_train_val, X_test, y_train_val, y_test = train_test_split(data, targets, test_size=0.2, random_state=2000)
	X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.4, random_state=2000)
	return X_train,y_train,X_val,y_val,X_test,y_test


def splitBatch(data,targets,BATCH_SIZE):
        sample_num=data.shape[0]
        batch_num=sample_num//BATCH_SIZE
        _data=[]
        _targets=[]
        for batch in range(batch_num):
            batch_data=[]
            batch_targets=[]
            for i in range(BATCH_SIZE):
                batch_data.append(data[batch*BATCH_SIZE+i])
                batch_targets.append(targets[batch*BATCH_SIZE+i])
            _data.append(batch_data)
            _targets.append(batch_targets)

        return torch.tensor(_data),torch.tensor(_targets,dtype=torch.long)


class makeDataset(Dataset):
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels
        self.n_samples=data.shape[0]
        
    def __getitem__(self,idx):
        return self.data[idx],self.labels[idx]

    
    def __len__(self):
        return len(self.n_samples)
    
