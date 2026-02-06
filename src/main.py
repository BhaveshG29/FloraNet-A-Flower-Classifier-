import torch as t
from torch.utils.data import DataLoader, Dataset
from model import FlowerCNN
import os
from dataset_builder import data_transform
from PIL import Image

#Importing the Test Dataset
class ImageDataset(Dataset):
    def __init__(self, folder, transform):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png",".jpg",".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

#Loading the Model and its saved_state_dictionary of weights
t.manual_seed(242)
model = FlowerCNN() 
saved_state_dict = "../cache/model.pth"
checkpoint = t.load(saved_state_dict, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint)


def test_model(path):
    test_dataset = ImageDataset(path, data_transform(train=False))
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False) #Creating DataLoader

    y_hist = [] #Saves all the prediction Ids

    #Evaluation Loop
    model.eval()
    with t.inference_mode():
        for X in test_dataloader:
            y_preds = model(X)
            preds = y_preds.argmax(dim=1)
            y_hist.extend(preds.tolist())

    #Loads the Idx to name JSON File
    import json
    with open("../data/flowers-102/cat_to_name.json", "r") as f:
        class_idx_to_names = json.load(f)

    class_names = {int(u):str(v) for u,v in class_idx_to_names.items()} #Convert str:str format to int:str format

    predicted_names = [class_names[idx+1] for idx in y_hist] #Converts the y_preds to Predicted Names of the Image

    #Prining the Predicted Names
    for i, name in enumerate(predicted_names):
        print(f"Image {i+1}: {name}")       



if __name__ == "__main__":
    path = "../input/"
    test_model(path=path)






