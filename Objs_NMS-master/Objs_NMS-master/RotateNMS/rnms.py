# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np
from RotateNMS.rotateiou import coordinate_convert_r2, intersection

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

""" 读取XML文件中的数据
"""
img_path = "rotate.jpg"
xml_path = "rotate.xml"

tree = ET.ElementTree(file=xml_path)

l_objs = []
l_scores = []
l_coors = []

for elem in tree.iter(tag='object'):
    name_str = str(elem[1].text)
    name_l = name_str.split("_")
    obj = name_l[0]
    score = float(name_l[1]) / 100.
    l_objs.append(obj)
    l_scores.append(score)

    bndbox = elem[5]
    cx = bndbox[0].text
    cy = bndbox[1].text
    w = bndbox[2].text
    h = bndbox[3].text
    ang = bndbox[4].text
    l_coors.append([cx, cy, w, h, ang])

l_objs = np.array(l_objs)
l_scores = np.array(l_scores, dtype=float)
l_coors= coordinate_convert_r2(np.array(l_coors, dtype=float).reshape(-1, 5))


""" 在NMS算法前的情况
"""
# before rnms
dst_img = cv2.imread(img_path)
for i in range(len(l_objs)):
    obj_name = l_objs[i]
    obj_coor = l_coors[i]
    coord = np.array([[l_coors[i][0], l_coors[i][1]], [l_coors[i][2], l_coors[i][3]],
                      [l_coors[i][4], l_coors[i][5]], [l_coors[i][6], l_coors[i][7]]], np.int32)
    cv2.polylines(dst_img, [coord], True, (0,255,255), 2)
cv2.imwrite("before.jpg", dst_img)

""" 在NMS算法之后的情况
"""
# after rnms
dst_img = cv2.imread(img_path)
iou_thread = 0.5
nms_objs = []
nms_scores = []
nms_coors = []

score_index = np.where(l_scores > 0.6)
l_objs = l_objs[score_index]
l_scores = l_scores[score_index]
l_coors = l_coors[score_index]


while len(l_objs) > 0:
    #--------------------------------------------------------------
    # 取出当前总集合方框集合中最大的方框，当前总集合最大的方框一定是保留的，放入nms_*中
    #--------------------------------------------------------------
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
    # 将当前最大框和其他所有的框比较旋转IOU，如果IOU大于0.5，就记录为删除的方框
    # ------------------------------------------------------------
    delete_l = []
    for j, select_coor in enumerate(l_coors):
        iou = intersection(max_coor, select_coor)
        if iou > iou_thread:
            delete_l.append(j)

    # ------------------------------------------------------------
    # 删除当前总集合中和最大框IOU大于0.5的框，更新总的集合框，这样总的集合框就
    # 越来越少，下一次取得最大值是更新后的总集合框里面的最大值
    # ------------------------------------------------------------
    l_objs = np.delete(l_objs, delete_l)
    l_scores = np.delete(l_scores, delete_l)
    l_coors = np.delete(l_coors, delete_l, axis=0)

for i in range(len(nms_objs)):
    obj_name = nms_objs[i]
    obj_coor = nms_coors[i]
    coord = np.array([[obj_coor[0], obj_coor[1]], [obj_coor[2], obj_coor[3]],
                      [obj_coor[4], obj_coor[5]], [obj_coor[6], obj_coor[7]]], np.int32)
    cv2.polylines(dst_img, [coord], True, (0,255,255), 2)
cv2.imwrite("after.jpg", dst_img)