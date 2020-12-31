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
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

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

    def validation_epoch_end(self, outputs):
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
        # Training Phase
        model.train()

        train_losses = []

        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return training_dataset.classes[preds[0].item()]


"""
----------- RESNET9 ---------------
"""


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]

    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)

        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)

        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(2),
                                        nn.Flatten(),
                                        nn.Linear(512 * 15 * 15, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)

        out = self.res1(out) + out

        out = self.conv3(out)
        out = self.conv4(out)

        out = self.res2(out) + out

        out = self.classifier(out)
        return out

def training():
    model_resnet = to_device(ResNet9(3, 5), device)  # num_classes=5
    num_epochs = 1
    otp_func = torch.optim.Adam
    lr = 0.001
    history3 = fit(num_epochs, lr, model_resnet, train_dl, val_dl, otp_func)
    PATH = './weather.pth'
    torch.save(model_resnet.state_dict(), PATH)
    return history3

def main():
    # show_batch(train_dl)
    training()


if __name__ == "__main__":
    main()
