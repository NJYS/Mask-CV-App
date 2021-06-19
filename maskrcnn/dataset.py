import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2

from utils import *


class MaskDataset(Dataset):
    def __init__(self, root='data', mode='train', transform=None):
        self.root = os.path.join(root, mode+'.csv')
        self.transform = transform
        self.dataframe = pd.read_csv(self.root)
        
    def __getitem__(self, idx):
        # load images ad masks
        line = self.dataframe.iloc[idx]

        img_path = line['path']
        mask_path = img_path[:-3] + 'png'
        img_cls = line['class'] + 2
        img = load_image(img_path)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        boxes = []

        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes[0] = list(map(float, line['bbox'].split()))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8) #.png에서 픽셀이 없는 친구들 -> 다 뺌

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels 
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.dataframe)
    
    
def get_DataLoader(root:str, mode = 'train', transform = None, 
                    batch_size=4, shuffle=True, num_workers=2, ratio=0.2):
    
    dataset = MaskDataset(root=root, mode=mode, transform=transform)
    
    n = len(dataset)
    
    if mode == 'test':
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, collate_fn=collate_fn,
                        pin_memory=True, drop_last=False)
        return dataset, test_loader
            
    else:        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, collate_fn=collate_fn,
                        pin_memory=True, drop_last=False)
        
        return dataset, loader

