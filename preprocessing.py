import re
import numpy as np


def getRidOfXXX(data_texts,model):
    new_one=[]
    for text in data_texts:
        list_tokens = []
        new_text=re.sub("[':!`\?,\.\)\(]",'',text.lower())
        
        word_list=new_text.split()
              
        for word in word_list:
          if word not in model:
            word_list.remove(word)
          else:
            list_tokens.append(word)
          
        new_text= ' '.join(list_tokens)
          
        new_one.append(new_text)
        
    return new_one


def getMaxLen(data_texts):
    max=0
    for text in data_texts:
        if len(text.split()) > max:
            max=len(text.split())
    return max



def getDic(data_texts_1,model):
    dic={}
    for text in data_texts_1:
        word_list=text.split()
        for word in word_list:
          if word not in model:
            #break
            print(word)
          elif word not in dic:
            dic[word]=model[word]

    EMBEDDING_SIZE = dic['what'].shape[0]
    dic['<PAD>']=np.array([0]*EMBEDDING_SIZE)

    return dic,EMBEDDING_SIZE


def padding(data_texts_1,max_len):
    new_data=[]
    for text in data_texts_1:
        delta=max_len - len(text.split())
        #print(delta)
        new_text='<PAD> '*delta+text
        new_data.append(new_text)
    return new_data
