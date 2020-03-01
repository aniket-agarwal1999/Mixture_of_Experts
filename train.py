import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import utils
from model import MoE


def train(args):

    if args.dataset == 'mnist':
        output_classes = 10

    transform = utils.get_transformation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)

    train_loader = DataLoader(dataset['train_data'], batch_size = args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset['val_data'], batch_size = args.batch_size, shuffle=True)

    model = MoE(args.num_experts, output_classes)

    model = utils.cuda(model, args.gpu_ids)

    for i in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            
            images, labels = utils.cuda([images, labels], args.gpu_ids)

            prediction = model(images)
            
