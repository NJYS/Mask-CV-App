import argparse
import os
from importlib import import_module

import torch
from torch.optim.lr_scheduler import StepLR

from utils import *
from dataset import *
import transforms as T
from engine import train_one_epoch, evaluate


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train(args):

    seed_everything(args.seed)
    args.name = args.name.replace(' ','_')
    saved_dir = f'saved/{args.model}_{args.name}'
    if os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    # -- settings
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # -- transform
    train_transform = get_transform(False)

    # -- dataset
    train_dataset, train_loader, val_dataset, val_loader = get_DataLoader(root=args.dataset, 
                                                                          mode = 'train', 
                                                                          transform = train_transform, 
                                                                          batch_size=args.batch_size, 
                                                                          shuffle=args.shuffle, 
                                                                          num_workers=args.num_workers)
    
    
    print(f'train_data {len(train_dataset)}, val_dataset {len(val_dataset)} loaded')

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    
    model = model_module(num_classes=args.num_classes).to(device)
    
    # -- loss & metric
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                            momentum=0.9, weight_decay=0.0005)

    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)


    print('Start training..')

    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1000)

        scheduler.step()

        evaluate(model, val_loader, device=device)

        file_name = f'epoch_{epoch}.pt'

        save_model(model, saved_dir, file_name)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 1)')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--dataset', type=str, default='dataframe.csv', help='dataset directory')
    parser.add_argument('--num_classes', type=int, default=13, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training (default: 8)')
    parser.add_argument('--model', type=str, default='mask_rcnn', help='model type (default: mask_rcnn)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate scheduler deacy step (default: 5)')
    parser.add_argument('--name', type=str, default='Baseline', help='model save at')

    
    # Container environment
    args = parser.parse_args()
    print(args)

    train(args)