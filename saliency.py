from imports import *
# from training import INIT_LR
from model import *
from dataloader import *

NUMBER = 14
INIT_LR = 1e-3

file = h5py.File("data/US_train.h5", "r+")
X_train = np.array(file["/images"])
y_train = np.array(file["/meta"])
file.close()
original_image = torch.tensor(X_train[NUMBER,:,:,:]).to(torch.float)
saved_img = X_train[NUMBER,:,:,:]
#reshape to 1,200,200
GLOBAL_IMAGE = original_image[None, :, :, :]

# model = NiN(lr=INIT_LR).to(device)
# model.load_state_dict(torch.load('MODEL/model.pth'))

model = torch.load("model/model.pth")
model.eval()

#test if model is correct by predicting
image = GLOBAL_IMAGE.to(device)
image.requires_grad_()
print(image.shape)
scores = model(image)

labels = y_train
labels = labels[NUMBER]
print("Predicted: ", scores)
print("Actual: ", labels)


# Get the index corresponding to the maximum score and the maximum score itself.
score_max_index = scores.argmax()
score_max = scores[0,score_max_index]

score_max.backward()

saliency, _ = torch.max(image.grad.data.abs(),dim=1)

# plt.subplot(1,2,1)
fig, ax = plt.subplots()
# code to plot the saliency map as a heatmap
ax.imshow(np.moveaxis(saved_img, 0, -1) / saved_img.max(), cmap='hot')
ax.imshow(torch.Tensor.cpu(saliency[0]), alpha=0.9, cmap='hot')
ax.axis('off')
ax.set_title('Saliency map, incorrect prediction, Class 3')
# plt.axis('off')
# plt.suptitle('Saliency map, incorrect prediction')
# plt.savefig('saliency.png')

# # plt.subplot(1,2,2)
# plt.imshow(saved_img)
# plt.axis('off')
# #title for both
plt.show()