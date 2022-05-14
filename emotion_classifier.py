"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from torch.autograd import Variable

import FacialExpressionRecognition.transforms as transforms
from skimage import io
from skimage.transform import resize
from FacialExpressionRecognition.models import *
from detect_faces import get_face_image_list

import cv2

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# list of every second for a video...that list will have the array of images
# one interation of a second we get say 7 photos we can


# raw_img = io.imread('FacialExpressionRecognition/images/1.jpg')

""" Using the function to get the faces - Shad """
vid_file_path = './dataset/00002.mp4'
face_list = get_face_image_list(vid_file_path)

# uncomment to show image
# cv2.imshow('image window', raw_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

net = ResNet18()
# net = VGG('VGG11')
'''vgg19 model'''
# net = VGG('VGG19')
checkpoint = torch.load(os.path.join('expressionmodels', 'resnet18_pt_mcn.pth'))
# checkpoint = torch.load(os.path.join('expressionmodels', 'vgg11_pt_mcn.pth'))
# checkpoint = torch.load(os.path.join('expressionmodels', 'PrivateTest_model.t7'))
net.load_state_dict(checkpoint, strict=False)
# net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

emotion_list = []
for list_second in face_list:
    emotion = [0] * 7
    for raw_img in list_second:
        gray = rgb2gray(raw_img)
        try:
            gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
        except ValueError:
            continue

        img = gray[:, :, np.newaxis]

        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = transform_test(img)

        ncrops, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cuda()
        # inputs = Variable(inputs, volatile=True)
        with torch.no_grad():
            inputs = Variable(inputs)
        outputs = net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg, dim=0)
        _, predicted = torch.max(outputs_avg.data, 0)

        emojis = str(class_names[int(predicted.cpu().numpy())])
        # print("The Expression is %s" % str(class_names[int(predicted.cpu().numpy())]))
        if emojis == "Angry":
            emotion[0] += 1
        elif emojis == "Disgust":
            emotion[1] += 1
        elif emojis == "Fear":
            emotion[2] += 1
        elif emojis == "Happy":
            emotion[3] += 1
        elif emojis == "Neutral":
            emotion[4] += 1
        elif emojis == "Sad":
            emotion[5] += 1
        elif emojis == "Surprise":
            emotion[6] += 1
    emotion_list.append(emotion)

# convert to df
df = pd.DataFrame(data=emotion_list,columns=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
df.to_csv('./output/resnet_frame_info2.csv') # save df as csv




