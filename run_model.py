import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import base64
from io import BytesIO
from imageio import imread
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from imantics import Mask
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_test_transform(height = 224, width = 224):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    return A.Compose([
                    A.Resize(height, width),
                    ToTensorV2()
                    ])

def load_image(img_encode):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    byte_img = base64.b64decode(img_encode)
    image = imread(BytesIO(byte_img))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.float32)
    image /= 255
    return image

def get_instance_segmentation_model(num_classes):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 64
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def post_processing(prediction):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    label_dict = {
    0:'',
    1:'wear_male_under25',
    2:'wear_male_over25',
    3:'wear_female_under25',
    4:'wear_female_over25',
    5:'incorrect_male_under25',
    6:'incorrect_male_over25',
    7:'incorrect_female_under25',
    8:'incorrect_female_over25',
    9:'NotWear_male_under25',
    10:'NotWear_male_over25',
    11:'NotWear_female_under25',
    12:'NotWear_female_over25',
    }
    result = {
        'number':0,
        'bboxes':{        
        },
        'labels':{
        },
        'segmentations':{
        }}
    threshold = 0.5
    
    for idx, (box, label, score, mask) in enumerate(zip(*prediction[0].values()), start=1):
        if score < threshold:
            result['number'] = idx-1
            break
        polygons = Mask((mask.detach().cpu().numpy() > 0.5).squeeze(0)).polygons()
        result['bboxes'].update({idx:[round(x,2) for x in box.detach().cpu().numpy().tolist()]})
        result['labels'].update({idx:label_dict[label.detach().cpu().numpy().tolist()]})
        polygon_list = ''
        for x, y in polygons.points[0].tolist():
            polygon_list += str(x)+','+str(y)+' '
        result['segmentations'].update({idx:polygon_list.rstrip()})
    else:
        result['number'] = idx
    return result

def inference_image(path,model):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    tfm = get_test_transform()
    #     image = load_image(test_paths[0])
    image = load_image(path)
    image = tfm(image=image)['image']

    with torch.no_grad():
        prediction = model(image.unsqueeze(0).to(device))

    return post_processing(prediction)

def GetBoundingBoxImage(file_path,info):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    fig,ax = plt.subplots(figsize=(15,15))
    ax.axis('off')
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(1024,1024))  
    
    for i in range(1,info['number']+1):
        xmin, ymin, xmax, ymax = map(int,info['bboxes'][i])
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
        
    ax.imshow(image)
    plt.imsave('./'+file_path.split('.')[0]+'_bbox.png',image)

def GetSegmentationImage(file_path,info):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    fig,ax = plt.subplots(figsize=(15,15))
    ax.axis('off')
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(1024,1024))  
    
    for i in range(1,info['number']+1):
        pixels = [j for j in zip(*info['segmentations'][i])]
        plt.fill(pixels[0], pixels[1],alpha=0.3)
        
    plt.imshow(image)
    plt.savefig('./'+file_path.split('.')[0]+'_seg.png')    

    
def GetBboxSegImage(file_path,info):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    fig,ax = plt.subplots(figsize=(15,15))
    ax.axis('off')
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(1024,1024))  
    
    for i in range(1,info['number']+1):
        xmin, ymin, xmax, ymax = map(int,info['bboxes'][i])
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
    
    for i in range(1,info['number']+1):
        pixels = [j for j in zip(*info['segmentations'][i])]
        plt.fill(pixels[0], pixels[1],alpha=0.3)
    
    ax.imshow(image)
    plt.savefig('./'+file_path.split('.')[0]+'_bbox_seg.png')
    