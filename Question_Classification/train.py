import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from dataloader import loadData,makeDataset
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from model import RNN,LSTM
from utils import encodingData

data,labels,EMBEDDING_SIZE=loadData()   
print("Loading...")
BATCH_SIZE=20  
#data_,labels_=splitBatch(data,labels,BATCH_SIZE)     
X_train,y_train,X_val,y_val,X_test,y_test=splitData(data,labels)
train_dataset = makeDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_dataset = makeDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
model =  LSTM(input_size=300, output_size=6, hidden_dim=64, n_layers=2)
MODEL_SAVE_PATH = './rnn_model.pt'

lr = 0.0001
N_EPOCHS = 200
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)


# train
train_losses = []
val_losses = []
best_val_loss = 1000

for epoch in range(N_EPOCHS):
    print('\nEpoch {}: '.format(epoch + 1))

    train_loss = []
    for X_train_batch, y_train_batch in tqdm(train_dataloader):
        out = model(X_train_batch)
        loss = loss_fn(out, y_train_batch)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_losses.append(sum(train_loss) / len(train_loss))

    val_loss = []
    for X_val_batch, y_val_batch in tqdm(val_dataloader):
        out = model(X_val_batch)
        loss = loss_fn(out, y_val_batch)
        train_loss.append(loss.item())
    val_losses.append(sum(train_loss) / len(train_loss))
    if best_val_loss > val_losses[-1]:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("Train loss: ", train_losses)
print("Validation loss: ", val_losses)

x = np.arange(len(train_losses))
fig, ax = plt.subplots()
ax.plot(x, train_losses, label='Train loss')
ax.plot(x, val_losses, label='Validation loss')
ax.legend()
plt.show()
