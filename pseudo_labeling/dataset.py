import os
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split

import cv2


class MaskDataLoader(Dataset):
    '''
    jpg and png should be a set of image and mask
    
    example >>> (1.jpg, 1.png) : png file is mask of jpg
    '''
    def __init__(self, root: str, mode = 'train', transform = None):
        super().__init__()
        
        self.img_paths = list(map(lambda x:os.path.join(root, x), 
                                  filter(lambda x:x[-3:] == 'jpg', os.listdir(root))))
        
        self.mode = mode
        self.transform = transform
        

        
    def __getitem__(self, index: int):
        
        
        # cv2 를 활용하여 image 불러오기
        image = cv2.imread(self.img_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        mask = cv2.imread(self.img_paths[index][:-3] + 'png')
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY).astype(bool).astype(np.uint8)
        
        
        if (self.mode in ('train', 'val')):
            
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

            return image, mask
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            
            return image
    
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.img_paths)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_DataLoader(root: str, mode = 'train', transform = None, 
                    batch_size=16, shuffle=True, num_workers=4, ratio=0.2):
    
    dataset = MaskDataLoader(root=root,  
                                mode=mode, transform=transform)
    
    n = len(dataset)
    
    if mode == 'test':
        drop_last = False
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