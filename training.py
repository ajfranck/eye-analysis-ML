from functions import *
from model import *
from dataloader import *
from imports import *

# define training hyperparameters
INIT_LR = 1e-5
EPOCHS = 50


model = NiN(lr=INIT_LR).to(device)

# initialize weights, requires forward pass for Lazy layers
X = next(iter(train_dataloader))[0].to(device)    # get a batch from dataloader
model.forward(X)                       # apply forward pass
model.apply(init_weights)              # apply initialization

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


plt.figure(figsize=(8,6))
train_losses = []
test_losses = []
test_accs_h = []
test_accs_v = []
for t in range(EPOCHS):
    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_acc_horizontal, test_acc_vertical = test_loop(val_dataloader, model, loss_fn)
    
    # plot
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accs_h.append(test_acc_horizontal/100)
    test_accs_v.append(test_acc_vertical/100)
    plt.clf()
    plt.plot(np.arange(1, t+2), train_losses, '-', label='train loss')
    plt.plot(np.arange(1, t+2), test_losses, '--', label='test loss')
    plt.plot(np.arange(1, t+2), test_accs_h, '-.', label='test acc horizontal')
    plt.plot(np.arange(1, t+2), test_accs_v, '-.', label='test acc vertical')
    plt.legend()
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.0001)
    
print(f"Final Accuracy Horizontal: {(test_acc_horizontal):>0.1f}%")
print(f"Final Accuracy Vertical: {(test_acc_vertical):>0.1f}%")
plt.savefig('accuracy_graph.png')

save_path = 'MODEL/model.pth'
torch.save(model.state_dict(), save_path)

