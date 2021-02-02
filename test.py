import torch
from sklearn.metrics import precision_score, f1_score, recall_score
from model import RNN
from dataloader import makeDataset
from torch.utils.data.dataloader import DataLoader
from train import X_test, y_test,BATCH_SIZE


model= RNN(input_size=300, output_size=6, hidden_dim=64, n_layers=1)
model.state_dict(torch.load(MODEL_SAVE_PATH))
test_dataset = makeDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print("Test results: ")
for X_test, y_test in test_dataloader:
    pred = torch.argmax(model(X_test), dim=1)
    print("Test precision: {}".format(precision_score(y_test, pred, average='weighted')))
    print("Test recall: {}".format(recall_score(y_test, pred, average='weighted')))
    print("Test F1-score: {}".format(f1_score(y_test, pred, average='weighted')))