# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division
from SoftNMS.softnmsfun import soft_nms_pytorch

import cv2
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

""" 读取XML文件中的数据
"""
img_path = "test.jpg"
xml_path = "test.xml"

tree = ET.ElementTree(file=xml_path)

l_objs = []
l_scores = []
l_coors = []

for elem in tree.iter(tag='object'):
    name_str = str(elem[0].text)
    name_l = name_str.split("_")
    obj = name_l[0]
    score = float(name_l[1]) / 100. 
    l_objs.append(obj)
    l_scores.append(score)

    bndbox = elem[4]
    xmin = bndbox[0].text
    ymin = bndbox[1].text
    xmax = bndbox[2].text
    ymax = bndbox[3].text
    l_coors.append([ymin,xmin,ymax,xmax])

l_objs = np.array(l_objs)
l_scores = np.array(l_scores, dtype=float)
l_coors = np.array(l_coors, dtype=float)

""" 在NMS算法前的情况
"""
# before nms
dst_img = cv2.imread(img_path)
for i in range(len(l_objs)):
    obj_name = l_objs[i]
    obj_coor = l_coors[i]
    cv2.rectangle(dst_img, (int(obj_coor[1]), int(obj_coor[0])), (int(obj_coor[3]), int(obj_coor[2])), (255,0,0), 2)
cv2.imwrite("before.jpg", dst_img)

""" 在NMS算法之后的情况
"""
# after nms
dst_img = cv2.imread(img_path)
iou_thread = 0.5
nms_objs = []
nms_scores = []
nms_coors = []

score_index = np.where(l_scores > 0.6)
l_objs = l_objs[score_index]
l_scores = l_scores[score_index]
l_coors = l_coors[score_index]

def cal_iou(box1, box2):
    xmin = np.maximum(box1[1], box2[1])
    ymin = np.maximum(box1[0], box2[0])
    xmax = np.minimum(box1[3], box2[3])
    ymax = np.minimum(box1[2], box2[2])

    s1 = (box1[3]-box1[1])*(box1[2]-box1[0])
    s2 = (box2[3]-box2[1])*(box2[2]-box2[0])

    w = np.maximum(xmax-xmin, 0.)
    h = np.maximum(ymax-ymin, 0.)
    inter = w * h
    union = np.maximum(s1+s2-inter,0.1)
    
    return inter / union

while len(l_objs) > 0:
    # --------------------------------------------------------------
    # 取出当前总集合方框集合中最大的方框，当前总集合最大的方框一定是保留的，放入nms_*中
    # --------------------------------------------------------------
    max_index = np.argmax(l_scores)
    max_obj = l_objs[max_index]
    max_score = l_scores[max_index]
    max_coor = l_coors[max_index]
    nms_objs.append(max_obj)
    nms_scores.append(max_score)
    nms_coors.append(max_coor)

    # ----------------------------------------
    # 从总集合l_*中删除当前总集合最大的方框
    # ----------------------------------------
    l_objs = np.delete(l_objs, [max_index])
    l_scores = np.delete(l_scores, [max_index])
    l_coors = np.delete(l_coors, [max_index], axis=0)

    # ------------------------------------------------------------
    # 将当前最大框和其他所有的框比较IOU，如果IOU大于0.5，就放入需要循环抑制的集合中
    # ------------------------------------------------------------
    delete_l = []
    press_coor = []
    press_coor.append(max_coor)
    press_score = []
    press_score.append(max_score)
    for select_coor, select_score, in zip(l_coors, l_scores):
        iou = cal_iou(max_coor, select_coor)
        if iou > iou_thread:
            press_coor.append(select_coor)
            press_score.append(select_score)
    no_keep = soft_nms_pytorch(press_coor, press_score)
    delete_l.append(no_keep)
    l_objs = np.delete(l_objs, delete_l)
    l_scores = np.delete(l_scores, delete_l)
    l_coors = np.delete(l_coors, delete_l, axis=0)


for i in range(len(nms_objs)):
    obj_name = nms_objs[i]
    obj_coor = nms_coors[i]
    cv2.rectangle(dst_img, (int(obj_coor[1]), int(obj_coor[0])), (int(obj_coor[3]), int(obj_coor[2])), (255,0,0), 2)

cv2.imwrite("cafter.jpg", dst_img)