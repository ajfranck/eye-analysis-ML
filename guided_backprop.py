from imports import *
from dataloader import *
from model import *
import matplotlib as mpl

# model = NiN(lr=0.01).to(device)
# model.load_state_dict(torch.load('model/model.pth'))
# model.eval()

model = torch.load("model/model.pth")
model.eval()

def hook_function(module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.),)

for i,module in enumerate(model.shared.modules()):
      if isinstance(module, nn.ReLU):
            module.register_full_backward_hook(hook_function)


f, axes = plt.subplots(5,6, figsize=(6,6))
axes = axes.reshape(-1)

f2, axes2 = plt.subplots(5,6, figsize=(6,5))
axes2 = axes2.reshape(-1)

for i,ax in enumerate(axes):

    if np.mod(i,5)==0:
        X,y = next(iter(val_dataloader))
        X.requires_grad_()
        X.retain_grad()

    ii = np.mod(i,5)
    # print(X.shape)
    # print(X[ii,:,:,:].unsqueeze(0).shape)
    # out = model(X[ii,:,:,:].unsqueeze(0).to(device))
    out = model(X.to(device))
    # print(out)
    #define out for horizontal only
    out = out[0,0]
    # print(out)
    out.backward()
    # bh = bin_data(out.cpu().detach())
    grad = X.grad[ii,0,:,:].cpu().numpy() # This is the guided backprop image
    print(grad)
    grad = grad/np.std(grad)

    ax.imshow(grad,cmap=mpl.colormaps["bwr"],vmin=-3,vmax=3)
    # ax[i].axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    # if b.numpy()[ii]==bh[0,0]:
    #   ax.patch.set_edgecolor('green')
    #   ax.set_title(f"{b.numpy()[ii]}")
    # elif np.abs(b.numpy()[ii]-bh[0,0])==1:
    #   ax.patch.set_edgecolor('orange')
    #   ax.set_title(f"{b.numpy()[ii]}({bh[0,0]})")
    # else:
    #   ax.patch.set_edgecolor('red')
    #   ax.set_title(f"{b.numpy()[ii]}({bh[0,0]})")

    ax.patch.set_linewidth(5)


    axes2[i].imshow(X[ii,0,:,:].detach().numpy(),cmap=mpl.colormaps["viridis"])
    # ax[i].axis('off')
    axes2[i].set_xticks([])
    axes2[i].set_yticks([])

plt.tight_layout()
f.savefig("gbp.pdf", format="pdf", bbox_inches="tight")
plt.show()