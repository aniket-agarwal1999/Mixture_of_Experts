import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim

import utils
from model import MoE


def train(args):

    if args.dataset == 'mnist':
        output_classes = 10

    transform = utils.get_transformation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)

    train_loader = DataLoader(dataset['train_data'], batch_size = args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset['val_data'], batch_size = args.batch_size, shuffle=True)

    CE_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = MoE(args.num_experts, output_classes)

    model = utils.cuda(model, args.gpu_ids)

    for i in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            
            images, labels = utils.cuda([images, labels], args.gpu_ids)

            optimizer.zero_grad()
            prediction = model(images)

            loss = CE_loss(prediction, labels)
            loss.backward()

            optimizer.step()
        
        if i%10 == 0:
            model.eval()
            for images, labels in val_loader:
                images, labels = utils.cuda([images, labels], args.gpu_ids)

                prediction = model(images)
                pred = prediction.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
            
            print('Epoch: ', i+1, ' Done!!    Accuracy: ', correct / len(val_loader.dataset))
        
        print('Epoch: ', i+1, ' Done!!    Loss: ', loss)

    torch.save(model.state_dict(), args.checkpoint_loc)
            
