from imports import *
from model import *
from dataloader import *


model = torch.load("model/model.pth")
model.eval()

#test if model is correct by predicting
dataloader = train_dataloader2

with torch.no_grad():
    for X, y in dataloader:
        #save the labels
        X = X.to(device)
        pred = model(X)
        pred = torch.Tensor.cpu(pred)
        y = torch.Tensor.cpu(y)
        # print(pred)
        # print('____')
        # print(y)

#graph scatterplot of horizontal labels and predictions vs vertical labels and predictions
plt.scatter(pred[:,0], pred[:,1], label = "preds", marker='+')  
plt.scatter(y[:,0], y[:,1], label = "labels", marker='o')
plt.xlim(1.5,3)
plt.ylim(1.5,3)
plt.legend()
plt.savefig('scatterplot.png')
plt.show()