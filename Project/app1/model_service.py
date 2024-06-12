# myapp/model_service.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import torchxrayvision as xrv
import numpy as np
import cv2

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")

    def forward(self, x):
        x = self.model(x)
        return x

def load_model():
    model = MyModel()
    MODEL_PATH = 'app1/model.pth'
    model = torch.load(MODEL_PATH)
    return model

def resize_image(input_path, target_size):
    image = Image.open(input_path)
    resized_image = image.resize(target_size)
    resized_array = np.array(resized_image)
    if len(resized_array.shape) == 3 and resized_array.shape[2] == 3:
        resized_array = resized_array[:, :, 0]
    return resized_array

def predict(image_path):
    input_path = image_path
    image = cv2.imread(input_path)
    target_size = (2000, 2000)

    target_size1 = (1500, 100)
    if image.shape[0] < target_size1[0] or image.shape[1] < target_size1[1]:
        image = resize_image(input_path, target_size)

    img = np.array(image)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = img[:, :, 0]

    img = xrv.datasets.normalize(img, 255)

    img = img[None, ...]
    transform = transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    img = transform(img)
    img = torch.from_numpy(img)

    # model = torch.load("app1/model.pth")

    # with torch.no_grad():
    #     outputs = model(img[None, ...])

    # return outputs
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.load_state_dict(torch.load('app1/model1.pth'))
    model.eval()
    outputs = model(img[None, ...])

    return outputs


