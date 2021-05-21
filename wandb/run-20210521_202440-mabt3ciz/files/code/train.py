from tqdm import tqdm
import torch
import wandb
import torch.nn as nn
from torchvision import models
from data_loading.data_loader import *
from data_loading.transforming import *

device = torch.device("cuda")
loaded_data = load_all_data()
EPOCHS = 100
model = models.resnet18().to(device)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 37)
model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
PROJECT_NAME = "Face-Mask-Detection-PT-V2"
wandb.init(project=PROJECT_NAME, name="testing")
for _ in tqdm(range(EPOCHS), leave=False):
    for data, label in tqdm(loaded_data, leave=False):
        data = data.to(device)
        label = label.to(device)
        preds = model(data.float())
        preds = preds.to(device)
        loss = criterion(preds, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    wandb.log({"loss": loss.item()})
wandb.finish()
