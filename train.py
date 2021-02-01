
import torch
import numpy as np
from torch import nn
#from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, f1_score, recall_score
from dataloader import loadData,splitData,splitBatch,makeDataset


data,labels,EMBEDDING_SIZE=loadData()   
print("Loading...")
BATCH_SIZE=20  
#data_,labels_=splitBatch(data,labels,BATCH_SIZE)     
X_train,y_train,X_val,y_val,X_test,y_test=splitData(data,labels)

X_train_val, X_test, y_train_val, y_test = train_test_split(corpus, types, test_size=0.2, random_state=2000)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.4, random_state=2000)

train_dataset = makeDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_dataset = makeDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_dataset = makeDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))
MODEL_SAVE_PATH = './rnn_model.pt'
model = RNN(input_size=300, output_size=6, hidden_dim=64, n_layers=1)

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

    # val_loss = []
    # for X_val_batch, y_val_batch in tqdm(val_dataloader):
    #     out = model(X_val_batch)
    #     loss = loss_fn(out, y_val_batch)
    #     train_loss.append(loss.item())
    # val_losses.append(sum(train_loss) / len(train_loss))
    # if best_val_loss > val_losses[-1]:
    #     best_val_loss = val_losses[-1]
    #     torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("Train loss: ", train_losses)
#print("Validation loss: ", val_losses)

x = np.arange(len(train_losses))
fig, ax = plt.subplots()
ax.plot(x, train_losses, label='Train loss')
#ax.plot(x, val_losses, label='Validation loss')
ax.legend()
plt.show()

model.state_dict(torch.load(MODEL_SAVE_PATH))
