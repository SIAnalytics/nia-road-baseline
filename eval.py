import argparse
import json
import os

import numpy as np
import cv2
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_filenames(label_dataset_path):
    filenames = []
    for filename in os.listdir(label_dataset_path):
        if filename.endswith(".json"):
            filenames.append(filename[:-5])
    return filenames


def make_mask(class_2_id, size, label):
    mask = np.zeros([size[0], size[1]], dtype=np.uint8)
    for r in range(len(label["features"])):
        road = label["features"][r]["properties"]
        type_name = road["type_name"]
        if type_name not in class_2_id.keys(): continue
        temp = road["road_imcoords"].split(",")
        if len(temp) <= 1: continue
        coords = np.array([int(round(float(c))) for c in temp]).reshape(-1, 2)
        cv2.fillPoly(mask, [coords], class_2_id[type_name])
    return mask


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_class", help="Number of class output", type=int)
    parser.add_argument("--source", help="path of test datasets", default='dataset/valid')
    parser.add_argument("--name", help="name of predictions", default='dataset/valid')
    args = parser.parse_args()

    if args.n_class == 7:
        class_2_id = {"Mortorway":1, "Primary":2, "Secondary":3, "Tertiary":4, "Residential":5, "Unclassified":6, "background":0}
    elif args.n_class == 2:
        class_2_id = {"Mortorway":1, "Primary":1, "Secondary":1, "Tertiary":1, "Residential":1, "Unclassified":1, "background":0}
    elif args.n_class == 5:
        class_2_id = {"Mortorway":1, "Primary":2, "Secondary":2, "Tertiary":2, "Residential":3, "Unclassified":4, "background":0}
    elif args.n_class == 3:
        class_2_id = {"Mortorway":1, "Primary":2, "Secondary":2, "Tertiary":2, "Residential":2, "Unclassified":2, "background":0}
    else:
        raise AttributeError

    filenames = []

    root = args.source
    label_root = os.path.join(root, 'label')
    target_root = os.path.join('submits', args.name)

    filenames = get_filenames(label_root)

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    for filename in filenames:
        target_path = os.path.join(target_root, filename + '_pred_idx.png')
        pred = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        with open(os.path.join(label_root, filename + '.json'), "r") as jfile:
            meta = json.load(jfile)
        gt = make_mask(class_2_id, pred.shape[0:2], meta)

        pred = torch.IntTensor(pred).cuda()
        gt = torch.IntTensor(gt).cuda()

        intersection, union, target = intersectionAndUnionGPU(pred, gt, args.n_class)
        intersection_meter.update(intersection)
        union_meter.update(union)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    iou_class = iou_class.cpu().numpy()
    print(iou_class)
    mIoU = np.mean(iou_class)
    print(mIoU)


if __name__ == '__main__':
    main()
