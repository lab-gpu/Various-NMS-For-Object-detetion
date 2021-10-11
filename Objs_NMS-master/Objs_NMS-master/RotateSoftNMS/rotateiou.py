from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pc
import math
import cv2

def draw(boxes_list):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    plt.xlim(-400, 700)
    plt.ylim(-400, 700)
    # 设置坐标轴刻度
    my_x_ticks = np.arange(-400, 700, 100)
    my_y_ticks = np.arange(-400, 700, 100)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    for i in range(len(boxes_list)):
        ax1.add_patch(
            pc.Polygon(
                [[boxes_list[i][0], boxes_list[i][1]],
                 [boxes_list[i][2], boxes_list[i][3]],
                 [boxes_list[i][4], boxes_list[i][5]],
                 [boxes_list[i][6], boxes_list[i][7]]],
                color = 'red',
                fill = None
            )
        )
    plt.show()  # 显示在figure

def coordinate_convert_r2(boxes):
    convert_boxs = []
    for box in boxes:
        convert_box = []
        w, h = box[2:-1]
        theta = -box[-1]

        x_lu, y_lu = -w / 2, h / 2
        x_ru, y_ru = w / 2, h / 2
        x_ld, y_ld = -w / 2, -h / 2
        x_rd, y_rd = w / 2, -h / 2

        x_ld_ = math.cos(theta) * x_ld + math.sin(theta) * y_ld + box[0]
        y_ld_ = -math.sin(theta) * x_ld + math.cos(theta) * y_ld + box[1]

        x_lu_ = math.cos(theta) * x_lu + math.sin(theta) * y_lu + box[0]
        y_lu_ = -math.sin(theta) * x_lu + math.cos(theta) * y_lu + box[1]

        x_ru_ = math.cos(theta) * x_ru + math.sin(theta) * y_ru + box[0]
        y_ru_ = -math.sin(theta) * x_ru + math.cos(theta) * y_ru + box[1]

        x_rd_ = math.cos(theta) * x_rd + math.sin(theta) * y_rd + box[0]
        y_rd_ = -math.sin(theta) * x_rd + math.cos(theta) * y_rd + box[1]

        if np.sqrt((x_lu_ - x_ru_) ** 2 + (y_lu_ - y_ru_) ** 2) > np.sqrt((x_ru_ - x_rd_) ** 2 + (y_ru_ - y_rd_) ** 2):
            tempx = x_lu_
            tempy = y_lu_
            x_lu_ = x_ru_
            y_lu_ = y_ru_
            x_ru_ = x_rd_
            y_ru_ = y_rd_
            x_rd_ = x_ld_
            y_rd_ = y_ld_
            x_ld_ = tempx
            y_ld_ = tempy

        convert_box += [x_lu_, y_lu_, x_ru_, y_ru_, x_rd_, y_rd_, x_ld_, y_ld_]

        convert_boxs.append(convert_box)
    convert_boxs = np.array(convert_boxs)

    return convert_boxs

def intersection(g, p):
    g = g.copy()
    p = p.copy()
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union

def intersections(g, ps):
    g = g.copy()
    ps = ps.copy()
    iou = []
    for p in zip(ps):
        p = p[0]
        g_temp = Polygon(g[:8].reshape((4, 2)))
        p_temp = Polygon(p[:8].reshape((4, 2)))
        if not g_temp.is_valid or not p_temp.is_valid:
            return 0
        inter = Polygon(g_temp).intersection(Polygon(p_temp)).area
        union = g_temp.area + p_temp.area - inter
        if union == 0:
            return 0
        else:
            iou.append(inter/union)
    return iou


if __name__ == '__main__':
    # x,y,w,h,θ
    rect_1 = np.array([400, 250, 300, 160, 30]).reshape(-1, 5)
    rect_2 = np.array([300, 350, 300, 160, 15]).reshape(-1, 5)
    rect_11 = coordinate_convert_r2(rect_1)
    rect_22 = coordinate_convert_r2(rect_2)
    rect = np.vstack((rect_11, rect_22))
    draw(rect)
    iou1 = intersection(rect_11, rect_22)
    print(iou1)
