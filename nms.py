import numpy as np

def nms(dets, topk, nms_thresh):
    """
    :param dets: num_boxes * 5
                [conf, x1, y1, x2, y2]
    :param topk:
    :return:
    """
    x1 = dets[:, 1]  # (num_boxes, )
    y1 = dets[:, 2]  # (num_boxes, )
    x2 = dets[:, 3]  # (num_boxes, )
    y2 = dets[:, 4]  # (num_boxes, )
    score = dets[:, 0]  # (num_boxes, )
    areas = (x2 - x1) * (y2 - y1)  # (num_boxes, )
    score_idx = score.argsort()[::-1]  # 置信度从大到小排序后的box的idx, (num_boxes, )
    keep = []

    while score_idx.shape[0] > 0:
        i = score_idx[0]
        keep.append(i)
        if keep == topk:
            break
        xx1 = np.maximum(x1[i], x1[score_idx[1:]])
        yy1 = np.maximum(y1[i], y1[score_idx[1:]])
        xx2 = np.minimum(x2[i], x2[score_idx[1:]])
        yy2 = np.minimum(y2[i], y2[score_idx[1:]])

        intersection_w = np.maximum(0, xx2 - xx1)
        intersection_h = np.maximum(0, yy2 - yy1)
        intersection = intersection_h * intersection_w
        iou = intersection / (areas[i] + areas[score_idx[1:]] - intersection)
        indices = np.where(iou < nms_thresh)[0]
        score_idx = score_idx[indices + 1]
    return keep
