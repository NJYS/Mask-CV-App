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

def get_train_transform(height = 224, width = 224):
    return A.Compose([
                    A.Resize(height, width),
                    ToTensorV2()
                    ])

def load_image(img_encode):
    byte_img = base64.b64decode(img_encode)
    image = imread(BytesIO(byte_img))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.float32)
    image /= 255
    return image

def get_instance_segmentation_model(num_classes):
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
        polygons = Mask((mask.numpy() > 0.5).squeeze(0)).polygons()
        result['bboxes'].update({idx:box.numpy().tolist()})
        result['labels'].update({idx:label_dict[label.numpy().tolist()]})
        result['segmentations'].update({idx:polygons.points[0].tolist()})
    else:
        result['number'] = idx
    return result

def inference_image(path):
    tfm = get_train_transform()
    #     image = load_image(test_paths[0])
    image = load_image(path)
    image = tfm(image=image)['image']

    model = get_instance_segmentation_model(13)
    model.load_state_dict(torch.load('mask_rcnn_final_state_dict.pt', map_location='cpu'))
    model.eval()

    with torch.no_grad():
        prediction = model(image.unsqueeze(0).to('cpu'))

    return post_processing(prediction)
