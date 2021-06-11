import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from torchvision import transforms, utils, models
from torchvision.transforms import Resize, ToTensor, Normalize
import base64
from io import BytesIO

result_dict ={
    0:'마스크,남자,청년',
    1:'마스크,남자,중년',
    2:'마스크,남자,노년',
    3:'마스크,여자,청년',
    4:'마스크,여자,중년',
    5:'마스크,여자,노년',
    6:'부정확,남자,청년',
    7:'부정확,남자,중년',
    8:'부정확,남자,노년',
    9:'부정확,여자,청년',
    10:'부정확,여자,중년',
    11:'부정확,여자,노년',
    12:'안씀,남자,청년',
    13:'안씀,남자,중년',
    14:'안씀,남자,노년',
    15:'안씀,여자,청년',
    16:'안씀,여자,중년',
    17:'안씀,여자,노년',
}

class TestDataset(Dataset):
    def __init__(self, img_encode, transform):
        self.img_encode = img_encode
        self.transform = transform
        byte_img = base64.b64decode(img_encode)
        self.img_list = [BytesIO(byte_img)]
        
        
    def __getitem__(self, index):
        image = Image.open(self.img_list[index])
        
        if self.transform:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.img_list)

def run_model(image_encode):

    test_transform = transforms.Compose([
        transforms.CenterCrop((300,200)),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = TestDataset(image_encode, transform = test_transform)

    loader = DataLoader(
        test_dataset
    )
    
    # model = models.resnet18(pretrained=True)
    # model.fc = nn.Linear(in_features=512, out_features=18)
    # torch.save(model, PATH)


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # result_path = '/app/test_mask/mask/model.pth'
    # result_path = 'mask/model.pth'
    # model.load_state_dict(torch.load(result_path,map_location=device), strict=False)
    
    # local test
    # model = torch.load('model.pt')
    
    # heroku server
    model = torch.load('model.pth')
    # model.eval()
    # model.to(device)
    # torch.save(model, 'model.pt')

    for images in loader:
        with torch.no_grad():
            # images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
    
        return result_dict[pred.cpu().numpy()[0]]




