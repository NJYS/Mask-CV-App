import os
import random
from collections import defaultdict

import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from model import *


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark=True

    np.random.seed(seed)
    random.seed(seed)

    
def get_all_path(root):
    all_path_dict = defaultdict(list)
    _get_all_path(root, all_path_dict)
    return all_path_dict
    
def _get_all_path(root, all_path_dict):
    for file_name in os.listdir(root):
        path = os.path.join(root, file_name)
        if os.path.isdir(path):
            _get_all_path(path, all_path_dict)
        else:
            all_path_dict[path.split('.')[-1]].append(path)


def load_model(model, path, num_class, pretrained='imagenet'):
    model = DeepLabV3Plus(model, pretrained, 3, num_class)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

########################Augmentation##########################
def get_train_transform(height = 512, width = 512):
    return A.Compose([
                        A.Resize(height, width),
                        ToTensorV2()
                        ])
    
    
def get_val_transform(height = 512, width = 512):
    return A.Compose([
                    A.Resize(height, width),
                    ToTensorV2()
                    ])
########################Augmentation##########################


########################Images################################
def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255
    return image

def save_image(path, output):
    cv2.imwrite(path, output)
    
def resize_image(image, size=(224,224)):
    return cv2.resize(image, dsize=size, interpolation=cv2.INTER_AREA)
########################Images################################

    
def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def add_hist(hist, label_trues, label_preds, n_class):
    """
        stack hist(confusion matrix)
    """

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                        minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def save_model(model, saved_dir, file_name='best_model(pretrained).pt', save_limit=10):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    file_list = os.listdir(saved_dir)
    if len(file_list) >= 10:
        for file in sorted(file_list, reverse=True):
            if 'best' in file:
                continue
            else:
                path = os.path.join(saved_dir, file)
                os.remove(path)
                break
        else:
            print('cannot remove file')
    torch.save(model.state_dict(), output_path)
    

def show_images(img_paths:list, n:int, base_dir='data'):
    fig = plt.figure(figsize=(24, 500))

    assert (len(img_paths) > n), 'number of paths should larger than n'
    
    img_paths = cv2.cvtColor(img_paths[:n], cv2.COLOR_RGB2BGR)
    
    transform = get_train_transform()
    
    idx = 1
    
    for path in img_paths:
        image = load_image(path)
        
        transformed = transform(image=image)
        image = transformed["image"]        
        
        title = '_'.join(path.split('/')[-2:])
        
        image = image.permute(1,2,0)
        
        
        ax = fig.add_subplot(n,4,idx)
        ax.imshow(image)
        ax.set_title(f'{title} : image')
        
        idx += 1
        
        mask_path = path.split('.')[0] + '.png'
        mask = load_image(mask_path)
        
        ax = fig.add_subplot(n,4,idx)
        ax.imshow(image+mask*255)
        
        idx += 1
        ax.set_title(f'{title} : mask')

    plt.show()