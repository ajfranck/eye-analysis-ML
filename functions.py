from dataloader import *
from imports import *
import cv2

theta_deg = 60
transform1 = transforms.Compose([
    transforms.RandomRotation(theta_deg,interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(0.5),
])

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for i, (X, y) in enumerate(dataloader):
        y = y.type(torch.int64)
        X = transform1(X.to(device))
        y = y.to(device)
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
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            y = y.type(torch.int64)
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct
