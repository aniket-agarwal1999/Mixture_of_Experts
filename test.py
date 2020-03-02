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

def test(args):
    if args.dataset == 'mnist':
        output_classes = 10

    transform = utils.get_transformation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)

    test_loader = DataLoader(dataset['test_data'], batch_size = args.batch_size, shuffle=False)

    model = MoE(args.num_experts, output_classes)

    model = utils.cuda(model, args.gpu_ids)

    if (args.checkpoint_loc == None):
        print('Please specify a checkpoint location for the model !!!')
    
    ckpt = utils.load_checkpoint(args.checkpoint_loc)
    model.load_state_dict(ckpt)

    model.eval()
    for images, labels in test_loader:
        
        images, labels = utils.cuda([images, labels], args.gpu_ids)

        prediction = model(images)
        pred = prediction.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    print('Final accuracy of the model is: ', correct / len(test_loader.dataset))
