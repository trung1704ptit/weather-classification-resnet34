import os
import torch
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

train_transformations = transforms.Compose([
    transforms.Resize((255, 255)), # resize input images to 255,255
    transforms.ToTensor()
])

test_transformations = transforms.Compose([
    transforms.Resize((255,255)), # resize input images to 255,255
    transforms.ToTensor()
])

data_dir= "./dataset"

classes = os.listdir(data_dir + "/Training")

cloudy_files = os.listdir(data_dir + "/Training/cloudy")
print('No. of training examples for cloudy images:', len(cloudy_files))

# apply the train and test transformations
training_dataset = ImageFolder(data_dir + "/Training", transform=train_transformations)
testing_dataset = ImageFolder(data_dir + "/Testing", transform=test_transformations)

print(training_dataset.classes)

# viewing the images by matplotlib
def show_example(img, label):
    print('Label: ', training_dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0)) #change dimension from 3,255,255 to 255,255,3 for matplotlib
    plt.show()
# show_example(*training_dataset[700])



# splitting training dataset into train_ds and val_ds
random_seed = 42
torch.manual_seed(random_seed)

val_size = 250
train_size = len(training_dataset) - val_size

train_ds, val_ds = random_split(training_dataset, [train_size, val_size])
# print(len(train_ds),len(val_ds))



# Create data loaders for training  and validation, to load data in batches
from torch.utils.data.dataloader import DataLoader
batch_size = 16
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

# We can look at the batches of images from dataset using the 'make_grid' of torchvision
from torchvision.utils import make_grid
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1,2,0))
        plt.show()
        break

def get_default_device():
    """ Using GPU if available or CPU """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """ Move tensor(s) to chosen device """
    if isinstance(data, (list, tuple)):
        return [to_device() for x in data]
    return data.to(device, non_blocking = True)

class DeviceDataLoader():
    """ wrap a Dataloader to move data to a device """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """ yield a batch of data after moving it to device """
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """ return number of batch size """
        return len(self.dl)

device = get_default_device()

"""
Now we can wrap our training and validation data loaders using DeviceDataLoader for automatically transferring batches
of data to GPU (if available), and use to_device to move our model to GPU (if available)
"""
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)


"""
Let's define the model by extending an ImageClassificationBase class which contains helper methods for training
and validation
"""
import torch.nn as nn
import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    # get the accuracy of number preds correctly
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images) # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images) # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        acc = accuracy(out, labels) # Calculate accuracy
        return {
            'val_loss': loss.detach(), 'val_acc': acc
        }

    def validation_epod_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean() # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean() # Combine accuracies
        return {
            'val_loss': epoch_loss.item(),
            'val_acc': epoch_acc.item()
        }

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

"""
----------- Training the Model ----------------
"""
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training phase
        model.train()

        train_losses = []

        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # validataion phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result);
        history.append(result);
    return history



def main():
    # show_batch(train_dl)
    print(device)

if __name__ == "__main__":
    main()
