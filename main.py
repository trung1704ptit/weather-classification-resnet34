import os
import torch
import torchvision
from torch.utils.data import random_split
import splitfolders  # or import split_folders
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor

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

