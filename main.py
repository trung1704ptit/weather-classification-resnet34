import os
import torch
import torchvision
from torch.utils.data import random_split
import splitfolders  # or import split_folders
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










def main():
    # show_batch(train_dl)
    device = get_default_device()
    print(device)

if __name__ == "__main__":
    main()
