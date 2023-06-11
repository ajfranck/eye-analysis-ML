from dataloader import *
from imports import *
import cv2

theta_deg = 60
transform1 = transforms.Compose([
    # transforms.Resize((224,224))
    # transforms.RandomRotation(theta_deg,interpolation=transforms.InterpolationMode.NEAREST),
    # transforms.RandomHorizontalFlip(0.5),
])

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for i, (X, y) in enumerate(dataloader):
        y = y.type(torch.int64)
        X = transform1(X.to(device))
        y = y.to(device)
        y = y.type(torch.float)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y).to(torch.float64)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            loss, current = loss.item(), (i + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return(loss_fn(pred, y).item())
    

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, correct2 = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # y = y.type(torch.int64)
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_vert = pred[:,0]
            pred_horiz = pred[:,1]
            y_vert = y[:,0]
            y_horiz = y[:,1]
            print('predicted vertical', pred_vert)
            print('actual vertical', y_vert)
            for i in range(pred.shape[0]):
                if(pred_horiz[i] - y_horiz[i] < 0.1 and pred_horiz[i] - y_horiz[i] > -0.1):
                    correct += 1
                if(pred_vert[i] - y_vert[i] < 0.1 and pred_vert[i] - y_vert[i] > -0.1):
                    correct2 += 1
            # correct += (pred_horiz == y_horiz).type(torch.float).sum().item()
            # correct2 += (pred_vert == y_vert).type(torch.float).sum().item()
            # correct = 0

    test_loss /= num_batches
    correct /= size
    correct2 /= size
    print(f"Test Error: \n Accuracy Vert: {(100*correct):>0.1f}%, Accuracy Horiz: {(100*correct2):>0.1f}%, \n Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct, 100*correct2
