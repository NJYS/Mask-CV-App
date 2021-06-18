import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2

from utils import *


class MaskDataset(Dataset):
    def __init__(self, root='dataframe.csv', transform=None):
        self.root = root
        self.transform = transform
        self.dataframe = pd.read_csv(root)
        
    def __getitem__(self, idx):
        # load images ad masks
        line = self.dataframe.iloc[idx]
        
        img_path = line['path']
        mask_path = img_path[:-3] + 'png'
        img_cls = line['class'] + 1
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
        
        # there is only one class
        boxes = [list(map(float, line['bbox'].split()))]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        labels = torch.ones((num_objs,), dtype=torch.int64) * img_cls
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
    
    dataset = MaskDataset(root=root, transform=transform)
    
    n = len(dataset)
    
    if mode == 'test':
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, collate_fn=collate_fn,
                        pin_memory=True, drop_last=False)
        return dataset, test_loader
            
    else:
        train_set, val_set = random_split(dataset, [n-int(n*ratio), int(n*ratio)])
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, collate_fn=collate_fn,
                        pin_memory=True, drop_last=False)
        
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, collate_fn=collate_fn,
                        pin_memory=True, drop_last=False)
        
        return train_set, train_loader, val_set, val_loader