#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 20:30:12 2023

@author: zok
"""

import torch
from torch import nn,load
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.3,0.4,0.4,0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.425,0.415,0.405),(0.205,0.205,0.205))
])

dataset = datasets.ImageFolder("dataset", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def get_model():
    model = models.inception_v3(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False #Freezing all the layers and changing only the below layers
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.fc = nn.Sequential(nn.Flatten(),
                            nn.Linear(2048,128),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(128,6))
    model.aux_logits = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, criterion, optimizer

# for epoch in range(10):
#     for i, (images, labels) in enumerate(dataloader):
#         model, criterion, optimizer = get_model()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        

#     print(f"Epoch [{epoch+1}/{10}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

#     torch.save(model.state_dict(), "resume_classifier.pth")

