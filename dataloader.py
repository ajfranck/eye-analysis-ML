from imports import *
from torch import utils

class ImageDataset:
    def __init__(self, images, labels):
        self.y = torch.tensor(np.float32(labels))
        self.X = torch.tensor(np.float32(images))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, index):
        return self.X[index, :, :, :], self.y[index, :]
    
transform = transforms.Compose([
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# Open training data
file = h5py.File("data/US_train.h5", "r+")
X_train = np.array(file["/images"])
y_train = np.array(file["/meta"])
file.close()

# Open test data
file = h5py.File("data/US_test_features.h5", "r+")
X_test = np.array(file["/images"])
file.close()

train_idx = np.arange(0, len(X_train))
np.random.shuffle(train_idx)

#split into train and validation
TRAIN_SPLIT = 0.8
train_idx = train_idx[0:int(TRAIN_SPLIT*len(train_idx))]
valid_idx = train_idx[int(TRAIN_SPLIT*len(train_idx)):len(train_idx)]

# Create training and validation data
X_valid = X_train[valid_idx, :, :]
y_valid = y_train[valid_idx]
X_train = X_train[train_idx, :, :]
y_train = y_train[train_idx]

#normalize using transform
# X_train = transform(X_train)
# X_valid = transform(X_valid)
# X_test = transform(X_test)

#Convert all to tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_valid = torch.tensor(X_valid).float()
y_valid = torch.tensor(y_valid).float()



dataset_validation = ImageDataset(X_valid, y_valid)
dataset_train = ImageDataset(X_train, y_train)
dataset_test = ImageDataset(X_test, np.zeros(len(X_test)))

# print("Train size: ", len(dataset_train))
# print("Validation size: ", len(dataset_validation))

#get random sample from dataset
X, y = dataset_train[0]

#create dataloaders
BATCH_SIZE = 8

train_dataloader = torch.utils.data.DataLoader(
    dataset = dataset_train,
    batch_size = BATCH_SIZE,
    shuffle = True)

val_dataloader = torch.utils.data.DataLoader(
    dataset = dataset_validation,
    batch_size = BATCH_SIZE,
    shuffle = True)

test_dataloader = torch.utils.data.DataLoader(
    dataset = dataset_test,
    batch_size = 1,
    shuffle = True)