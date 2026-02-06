import torch as t 
from torch import nn, optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from dataset_builder import FloraNet
from metrics import Accuracy, F1_score

#Getting the Train-Test Split Dataset from dataset_builder.py file
train_dataset, val_dataset = FloraNet() 

#Parameters for DataLoader
Batch_size = 128
n_woker = 10

#Creating train-test DataLoader with seed for the same randomness
t.manual_seed(124)
train_dataloader, val_dataloader = DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True, num_workers=n_woker, pin_memory=True), DataLoader(dataset=val_dataset, batch_size=Batch_size, shuffle=False, num_workers=n_woker, pin_memory=True)

#Convolutional Block OR Residual Block for Every Stage. 
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=stride, padding=1, bias=False), #Batch-Norm Already provides the Bias
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_c),
        )

        self.skip = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_c)
            )

    def forward(self, x):
        return t.relu(self.conv(x) + self.skip(x))

#Actual CNN for Classifing Labels
class FlowerCNN(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()

        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = ConvBlock(64, 64) #No Residual
        self.stage2 = ConvBlock(64, 128, stride=2) #Residual-1
        self.stage3 = ConvBlock(128, 256, stride=2) #Residual-2
        self.stage4 = ConvBlock(256, 512, stride=2) #Residual-3
        
        # Single Hidden Layer FNN with Dropout for Regularization
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(in_features=512, out_features= 512), #(512, 512)
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        x = self.stage0(x) 
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.classifier(x)



# Training Model
def train_model(model=FlowerCNN(), alpha=0.01, epochs=2, optim_type="SGD"):
    # Device Agnostic Code
    device = "cuda" if t.cuda.is_available() else "cpu" 
    model = model.to(device=device)
    scaler = GradScaler(device=device)

    optimizer = optim.SGD(model.parameters(), lr=alpha)

    if(optim_type.lower() == "adam"):
        optimizer = optim.Adam(params=model.parameters(), lr=alpha, betas=(0.9, 0.999), eps=1e-10, weight_decay=1e-4)

    elif(optim_type.lower() == "adamax"):
        optimizer = optim.Adamax(params=model.parameters(), lr=alpha, eps=1e-10)

    elif(optim_type.lower() == "rmsprop"):
         optimizer = optim.RMSprop(params=model.parameters(), lr=alpha, eps=1e-10)

    elif(optim_type.lower() == "adagrad"):
         optimizer = optim.Adagrad(params=model.parameters(), lr=alpha, eps=1e-10)
    
    elif(optim_type.lower() == "adamw"):
        optimizer = optim.AdamW(params=model.parameters(), lr=alpha, eps=1e-10, weight_decay=1e-4)
    
    #LR-Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)

    loss_func = nn.CrossEntropyLoss()

    loss_hist = [4.3210] #Assumed so that pbar.set_description shows something 
    acc_hist = []
    f1_hist = []

    from tqdm import tqdm
    pbar = tqdm(range(epochs)) # Progress-bar

    for epoch in pbar:
        pbar.set_description(f"Loss:{loss_hist[-1]:.4f}")
        if epoch == 0:
            loss_hist.pop(0) #Pops out the Random Intialized Value we assumed 


        train_loss = 0 #Per Batch Average
        train_acc = 0 #Accuracy Per Batch
        train_f1 = 0 #F1-score Per Batch
        
        model.train()
        
        for (X, y) in train_dataloader:
            X,y = X.to(device, non_blocking=True), y.to(device, non_blocking=True) # Allows Batch to be on GPU while CPU also keeps processing the next Batch
            
            optimizer.zero_grad(set_to_none=True) #Sets the Tensor to None instead of Zeros
            
            with autocast(device_type=device, dtype=t.float16): #Does some of the operations as float16 instead of float32. Faster Computation
                y_preds = model(X)

                loss = loss_func(y_preds, y)
            
            train_loss += loss.item()/len(train_dataloader)
            
            train_acc += Accuracy(y_preds, y)/len(train_dataloader)
            train_f1 += F1_score(y_preds, y, device=device)/len(train_dataloader)
            
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer=optimizer)
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer=optimizer)
            scaler.update()

        scheduler.step() 


        loss_hist.append(train_loss)
        acc_hist.append(train_acc)
        f1_hist.append(train_f1)

    return loss_hist, acc_hist, f1_hist


# Validating the Model
def validate_model(model=FlowerCNN()):
    device = "cuda" if t.cuda.is_available() else "cpu" 
    model = model.to(device=device)

    loss_func = nn.CrossEntropyLoss()
    
    model.eval().to(device=device)
    with t.inference_mode():
        val_loss = 0
        val_acc = 0
        val_f1 = 0

        for (X,y) in val_dataloader:
            X, y = X.to(device), y.to(device)
            
            with autocast(device_type=device, dtype=t.float16):
                y_preds = model(X)
                loss = loss_func(y_preds, y) 

            val_loss += loss.item()/len(val_dataloader) 
    
            val_acc += Accuracy(y_preds, y)/len(val_dataloader)
            val_f1 += F1_score(y_preds, y, device=device)/len(val_dataloader)

    return val_loss, val_acc, val_f1

#Saving the Model and its Metrics into desired Directories
def save_model(model=FlowerCNN(), save_metric=True):
    import time
    start = time.time()

    loss_hist, acc_hist, f1_hist = train_model(model=model, alpha=0.01, epochs=500, optim_type="AdamW")
    
    val_loss, val_acc, val_f1 = validate_model(model=model)

    end_time = time.time() - start #Total Time taken to Compute

    if save_metric:
        import json
        loss_hist.append(val_loss) #Last Value will be Validating Loss
        acc_hist.append(val_acc) #Last Value will be Validating Accuracy
        f1_hist.append(val_f1) #Last Value will be Validating F1 Score

        data = {
            "Loss": loss_hist,
            "Accuracy": acc_hist,
            "F1 Score": f1_hist,
            "Time": end_time
            }

        with open("../cache/metric.json", "w") as f:
            json.dump(data, f, indent=2)


    t.save(model.state_dict(), "../cache/model.pth")



if __name__ == "__main__":
    import time 
    start_time = time.time()

    model = FlowerCNN(102)
    save_model(model=model, save_metric=True)
    
    end_time = time.time() - start_time

    print(f"Total Time Taken: {end_time:.3f} seconds")
    
    # Plottint Curves and Creating Animation
    from visuals import visuals
    v = visuals()
    v.curve_plotter(show_plot=True)
    v.animate_loss(show_animation=False)
    v.animate_accuracy(show_animation=False)
    v.animate_f1_score(show_animation=False)
