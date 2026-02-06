import torch as t
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.transforms import RandAugment
from torchvision.datasets import ImageFolder
from typing import Tuple


#Wrapper Class to data_transform
class Data_Transform_Wrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.transform(img), label

    def __len__(self):
        return len(self.dataset)

#Transformations
def data_transform(train: bool = True):
    mean = [0.42575, 0.4012, 0.28805]
    std = [0.23945, 0.20345, 0.19975]

    if train:
        return transforms.Compose([
            #Data Augmentation for Training Datasets aka Better Generalization  
            transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
            # transforms.Resize((64,64)),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return transforms.Compose([
        transforms.Resize((96, 96)), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])




#Dataset Builder Function OR Returns the Dataset
def FloraNet() -> Tuple[Dataset, Dataset]:

    dataset1_train = ImageFolder(root="../data/flowers-102/flowers/train")
    dataset1_val = ImageFolder(root="../data/flowers-102/flowers/val")
    dataset1_test = ImageFolder(root="../data/flowers-102/flowers/test")

    train_dataset = dataset1_train #Train Dataset
    test_dataset = ConcatDataset([dataset1_test, dataset1_val]) #Concatenating Validate and Test Dataset

    train_dataset, test_dataset = Data_Transform_Wrapper(train_dataset, data_transform(train=True)), Data_Transform_Wrapper(test_dataset, data_transform(train=False))

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = FloraNet() #Testing if everything works


