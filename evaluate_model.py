from imports import *
from model import *
from dataloader import *


model = model(lr=0.01).to(device)
model.load_state_dict(torch.load('model/model.pth'))
model.eval()

#test if model is correct by predicting
dataloader = val_dataloader

size = len(dataloader.dataset)
num_batches = len(dataloader)
horizontal_pred = np.zeros(size)
vertical_pred = np.zeros(size)
horizontal_labels = np.zeros(size)
vertical_labels = np.zeros(size)

with torch.no_grad():
    for X, y in dataloader:
        #save the labels
        print(X.shape)
        print(y.shape)
        X = X.to(device)
        pred = model(X)
        print(pred)
        print('____')
        print(y)
        # for i in range(y.shape[1]): 
        #     horizontal_labels[i] = y[0,i]
        #     vertical_labels[i] = y[1,i]
        #     horizontal_pred[i] = pred[0,i]
        #     vertical_pred[i] = pred[1,i]


#print the labels
# print(horizontal_labels)
# print("_________")
# print(horizontal_pred)