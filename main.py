import os
import torch
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np

train_transformations = transforms.Compose([
    transforms.Resize((255, 255)), # resize input images to 255,255
    transforms.ToTensor()
])

test_transformations = transforms.Compose([
    transforms.Resize((255,255)), # resize input images to 255,255
    transforms.ToTensor()
])

data_dir= "./dataset"
PATH = './weather.pth'

classes = os.listdir(data_dir + "/Training")

cloudy_files = os.listdir(data_dir + "/Training/cloudy")
print('No. of training examples for cloudy images:', len(cloudy_files))

# apply the Training and test transformations
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # setup custom optimizer with wight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

    # setup one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()

        train_losses = []
        lrs = []

        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record and update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
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

class ResNet34CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=True)

        num_features = self.network.fc.in_features
        self.network.fc = nn.Linear(num_features, 5)

    def forward(self, xb):
        xb = self.network(xb)
        return xb

    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True

    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True

def training():
    pass

def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');

def load_model():
    model = torch.load(PATH)
    return model

def save_model(model):
    torch.save(model, PATH)

def main():
    # show_batch(train_dl)
    model_resnet34 = to_device(ResNet34CnnModel(), device)
    model_resnet34.freeze()

    epochs = 8
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    model_resnet34.unfreeze()

    epochs = 8
    history6 = fit_one_cycle(epochs, max_lr, model_resnet34, train_dl, val_dl,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=opt_func)

    model_resnet34.unfreeze()

    epochs = 8
    history6 += fit_one_cycle(epochs, 0.001, model_resnet34, train_dl, val_dl,
                              grad_clip=grad_clip,
                              weight_decay=weight_decay,
                              opt_func=opt_func)

    # history6 = training()

    plot_accuracies(history6)
    plot_losses(history6)
    plot_lrs(history6)
    plt.show()

    test_loader = DeviceDataLoader(DataLoader(testing_dataset, batch_size*2), device)
    result = evaluate(model_resnet34, test_loader)
    print(result)

    img, label = testing_dataset[76]
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    print('Label:', training_dataset.classes[label], ', Predicted:', predict_image(img, model_resnet34))

    img, label = testing_dataset[125]
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    print('Label:', training_dataset.classes[label], ', Predicted:', predict_image(img, model_resnet34))

if __name__ == "__main__":
    main()
