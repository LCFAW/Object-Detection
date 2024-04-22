from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
# from data import VOC_ROOT, VOC_CLASSES as labelmap
from data import Trail_ROOT, Trail_CLASSES as labelmap
from PIL import Image
# from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import TrailAnnotationTransform, TrailDetection, BaseTransform, Trail_CLASSES
import torch.utils.data as data
from ssd import build_ssd
import cv2



def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    if not os.path.exists('/home/zhaojiankuo/ssd.pytorch-master/img_out'):
        os.makedirs('/home/zhaojiankuo/ssd.pytorch-master/img_out')

    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                img_with_boxes = draw_boxes(img, coords, label_name, score)
                # 保存带有预测结果的图像
                save_path = os.path.join(save_folder, img_id + '.jpg')
                cv2.imwrite(save_path, img_with_boxes)
                j += 1

def draw_boxes(image, coords, label, score):
    # 绘制边界框
    top_left = (int(coords[0]), int(coords[1]))
    bottom_right = (int(coords[2]), int(coords[3]))
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)

    # 添加标签和分数
    label_text = "{}: {:.2f}".format(label, score)
    cv2.putText(image, label_text, (top_left[0], top_left[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image