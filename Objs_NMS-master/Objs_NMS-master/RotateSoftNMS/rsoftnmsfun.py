import torch
import numpy as np
def soft_nms_pytorch2(dets, box_scores, sigma=0.5, thresh=0.5, cuda=0):
    N = dets.shape[0]
    indexes = torch.arange(0, N, dtype=torch.int).view(N, 1)
    dets_tem = torch.cat((dets, indexes), dim=1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [x1,y1,x2,y2]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                # 如果当前的置信度小于后面的最大的置信度，那么就交换坐标，得分，还有面积
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate选出里面相交的四个点
        xx1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        yy1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        inter = inter.to(torch.float64)
        outer = (areas[i].to(torch.float64) + areas[pos:].to(torch.float64) - inter)
        ovr = torch.div(inter, outer)

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    mask = scores > thresh
    keep = dets[:, 4][mask]
    # 返回保留的边框的index
    keep_ind = keep.int()

    return keep_ind, dets_tem


def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.5, cuda=0):
    dets = np.array(dets)
    box_scores = np.array(box_scores)
    N = dets.shape[0]
    indexes = np.arange(0, N, dtype=np.int).reshape(-1, 1)
    dets = np.hstack((dets, indexes))


    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        tscore = scores[i]
        pos = i + 1

        #---------------------------------------------------
        # 拿出第pos后，所有box置信度最大的box，其实就是每次都要找最大置信度的box
        #---------------------------------------------------
        if i != N - 1:
            maxpos, maxscore = max(enumerate(scores[pos:]), key=lambda x: x[-1])

            if tscore < maxscore:
                # ---------------------------------------------------
                # 如果当前的置信度小于后面的最大的置信度，那么就交换坐标，得分，还有面积
                # ---------------------------------------------------
                dets[i], dets[maxpos + i + 1] = np.copy(dets[maxpos + i + 1]), np.copy(dets[i])
                scores[i], scores[maxpos + i + 1] = np.copy(scores[maxpos + i + 1]), np.copy(scores[i])
                areas[i], areas[maxpos + i + 1] = np.copy(areas[maxpos + i + 1]), np.copy(areas[i])

        # IoU calculate选出里面相交的四个点
        xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
        yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
        xx2 = np.minimum(dets[i, 2], dets[pos:, 2])
        yy2 = np.minimum(dets[i, 3], dets[pos:, 3])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        outer = (areas[i] + areas[pos:] - inter)
        ovr = np.divide(inter, outer)

        # ---------------------------------------------------
        # 采取高斯的方式衰减置信度
        # ---------------------------------------------------
        weight = np.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    mask = scores < thresh
    no_keep = dets[:, 4][mask].astype(np.int)

    return no_keep