import argparse
import os
from importlib import import_module
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import segmentation_models_pytorch as smp

from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import *
from dataset import *
from loss import *



def train(args):
    wandb.init(project='Mask_Seg_Pseudo_Labeling', name=f'{args.name}')
    wandb.config.update(args)

    seed_everything(args.seed)
    args.name = args.name.replace(' ','_')
    saved_dir = f'saved/{args.model}_{args.name}'

    # -- settings
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # -- transform
    train_transform = get_train_transform(height = args.image_resize, width = args.image_resize)
    val_transform = get_val_transform(height = args.image_resize, width = args.image_resize)

    # -- dataset
    train_dataset, train_loader, val_dataset, val_loader = get_DataLoader(root=args.dataset, mode = 'train', transform = train_transform, 
                    batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    
    
    print(f'train_data {len(train_dataset)}, val_dataset {len(val_dataset)} loaded')

    num_classes = args.num_classes

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    
    model = model_module(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=args.in_channels,
        classes=num_classes
    ).to(device)
    
    
    # -- loss & metric
    criterion = create_criterion(args.criterion)
    
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-6
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)


    print('Start training..')
    best_loss = np.Inf
    best_mIoU = 0
    for epoch in range(args.epochs):
        model.train()
        mean_loss = 0
        for step, (images, masks) in enumerate(train_loader):
            images = torch.stack(images).to(device)       # (batch, channel, height, width)
            masks = torch.stack(masks).long().to(device)  # (batch, channel, height, width)
            
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            mean_loss += loss
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, args.epochs, step+1, len(train_loader), loss.item()))

        wandb.log({'train_loss': mean_loss/(step+1)})    
        
        scheduler.step()
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % args.val_every == 0:
            avrg_loss, mIoU = validation(epoch + 1, model, val_loader, criterion, device, args)
            if avrg_loss < best_loss:
                if not os.path.isdir(saved_dir):
                    os.mkdir(saved_dir)

                print('[loss] Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_loss = avrg_loss
                save_model(model, saved_dir=saved_dir, file_name = f'epoch_{epoch}_loss_{best_loss:.4f}.pth', save_limit=args.save_limit)
            elif mIoU > best_mIoU:
                print('[mIoU] Best performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_mIoU = mIoU
                save_model(model, saved_dir, file_name = f'epoch_{epoch}_mIoU_{best_mIoU:.4f}.pth', save_limit=args.save_limit)


def validation(epoch, model, data_loader, criterion, device, args):
    print('Start validation #{}'.format(epoch))
    model.eval()
    hist = np.zeros((2, 2)) # 12 : num_classes
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        for step, (images, masks) in enumerate(data_loader):
            
            images = torch.stack(images).to(device)       # (batch, channel, height, width)
            masks = torch.stack(masks).long().to(device)  # (batch, channel, height, width)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            
            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=args.num_classes)
            
        acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)    
        
        wandb.log({'val_acc': acc, 'mIoU':mIoU})

        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}, acc : {:.4f}'.format(epoch, avrg_loss, mIoU, acc))

    return avrg_loss, mIoU



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 1)')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle')
    parser.add_argument('--num_workers', type=int, default=2, help='num_workers')
    parser.add_argument('--dataset', type=str, default='data', help='dataset directory')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size for training (default: 8)')
    parser.add_argument('--valid_batch_size', type=int, default=8, help='input batch size for validing (default: 8)')
    parser.add_argument('--val_every', type=int, default=1, help='validation every {val_every}')
    parser.add_argument('--model', type=str, default='DeepLabV3Plus', help='model type (default: DeepLabV3Plus)')
    parser.add_argument('--encoder_name', type=str, default='senet154', help='model encoder type (default: SeNet154)')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='model pretrain weight type (default: imagenet)')
    parser.add_argument('--in_channels', type=int, default=3, help='number of channels (default: 3)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate scheduler deacy step (default: 5)')
    parser.add_argument('--name', type=str, default='Baseline', help='model save at')
    parser.add_argument('--save_limit', type=int, default=10, help='maximum limitation to save')
    parser.add_argument('--image_resize', type=int, default=1024, help='resize image to train & val & test')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--name', default='Baseline Code', help='model save at')

  
    # Container environment
    args = parser.parse_args()
    print(args)

    # train(args)

    