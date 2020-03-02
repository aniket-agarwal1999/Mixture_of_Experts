import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, transforms, models

def cuda(xs, gpu_id):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda(int(gpu_id[0]))
        else:
            return [x.cuda(int(gpu_id[0])) for x in xs]
    return xs

def load_checkpoint(ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def get_dataset(dataset ,transform = None, train_split=0.8):

    if args.dataset == 'mnist':
        if transform == None:
            print('Specify a transformation function')
        
        training_data = datasets.MNIST('./mnist', train=True, transform = transform['train_transform'], download=True)
        num_training_imgs = int(len(training_data) * train_split)
        torch.manual_seed(0)
        train_data, val_data = torch.utils.data.random_split(training_data, [num_training_imgs, len(training_data) - num_training_imgs])

        test_data = datasets.MNIST('./mnist', train=False, transform = transform['test_transform'], download=True)

        return {'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data}


def get_transformation(dataset):
    
    if dataset == 'mnist':
        train_transform = transforms.Compose([transforms.RandomResizedCrop(size=28),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

        test_transform = transforms.Compose([transforms.Resize(size=28),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

        transform = {'train_transform': train_transform,
                     'test_transform': test_transform}

    else:
        print('Please choose the right dataset!!!')

    return transform


def EM_loss():
    '''
    I am still not able to get the notation specified in the paper for calculating EM loss as it is a bit confusing how they have
    calculated the posterior probabilities in equation (5).
    Also the updation step for the expert and gating weights they have used is just simply iteration based rather than gradient based which
    is also pretty weird.

    Also Cross Entropy would also work in this particular case as the model is a classification model and
    I think that the gradient based method also solves the MLE optimization problem(for which EM algorithm is suggested) as specified in the paper.
    '''
