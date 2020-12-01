import cv2
import numpy as np
import os
import osgeo
import glob
import json

import PIL
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision import transforms

class NIADataset(Dataset):
    def __init__(self, root, patch_size, shuffle=False, rgb=True, n_class=7):
        self.root = root
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.rgb = rgb
        self.labels = glob.glob(os.path.join(root, "label", "*.json"))
        self.samples = []
        self.stretch_val = 1.6
        self.mean_val = 0.5
        self.std_val = 0.5 / self.stretch_val

        self.allocate_clsss_book(n_class)

        for f in self.labels:
            img = os.path.join(root, "asset", os.path.splitext(os.path.basename(f))[0] + ".png")
            assert os.path.isfile(img)
            self.samples.append([f, img])
        self.samples.sort()
        if self.shuffle:
            np.random.shuffle(self.samples)
        self.preproc_manual = {"crop_size":(0.7, 1.0), "resize_p":0.5, "hflip_p":0.5, "vflip_p":0.5, "color_p":0.5}
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.1)
        self.normalize = transforms.Normalize(mean=(self.mean_val, self.mean_val, self.mean_val), std=(self.std_val, self.std_val, self.std_val))
        self.denormalize = transforms.Normalize(mean=(-self.stretch_val, -self.stretch_val, -self.stretch_val), std=(2*self.stretch_val, 2*self.stretch_val, 2*self.stretch_val))

    def allocate_clsss_book(self, n_class):
        self.class_names = ["Mortorway", "Primary", "Secondary", "Tertiary", "Residential", "Unclassified", "background", "Motorway"]
        if n_class == 7:
            self.colorbook = {"Mortorway":(51, 51, 255), "Primary":(51, 255, 255), "Secondary":(51, 255, 51), "Tertiary":(255, 255, 51), "Residential":(255, 51, 51), "Unclassified":(255, 51, 255), "background":(0, 0, 0)}
            self.class_2_id = {"Mortorway":1, "Primary":2, "Secondary":3, "Tertiary":4, "Residential":5, "Unclassified":6, "background":0}
            self.id_2_class = {1:"Mortorway", 2:"Primary", 3:"Secondary", 4:"Tertiary", 5:"Residential", 6:"Unclassified", 0:"background"}
        elif n_class == 2:
            self.colorbook = {"Mortorway":(255, 255, 255), "Primary":(255, 255, 255), "Secondary":(255, 255, 255), "Tertiary":(255, 255, 255), "Residential":(255, 255, 255), "Unclassified":(255, 255, 255), "background":(0, 0, 0)}
            self.class_2_id = {"Mortorway":1, "Primary":1, "Secondary":1, "Tertiary":1, "Residential":1, "Unclassified":1, "background":0}
            self.id_2_class = {1:"Road", 0:"background"}
        elif n_class == 5:
            self.colorbook = {"Mortorway":(51, 51, 255), "Primary":(51, 255, 255), "Secondary":(51, 255, 255), "Tertiary":(51, 255, 255), "Residential":(255, 51, 51), "Unclassified":(255, 51, 255), "background":(0, 0, 0)}
            self.class_2_id = {"Mortorway":1, "Primary":2, "Secondary":2, "Tertiary":2, "Residential":3, "Unclassified":4, "background":0}
            self.id_2_class = {1:"Mortorway", 2:"etcRoad", 3:"Residential", 4:"Unclassified", 0:"background"}
        elif n_class == 3:
            self.colorbook = {"Mortorway":(51, 51, 255), "Primary":(51, 255, 255), "Secondary":(51, 255, 255), "Tertiary":(51, 255, 255), "Residential":(51, 255, 255), "Unclassified":(51, 255, 255), "background":(0, 0, 0)}
            self.class_2_id = {"Mortorway":1, "Primary":2, "Secondary":2, "Tertiary":2, "Residential":2, "Unclassified":2, "background":0}
            self.id_2_class = {1:"Mortorway", 2:"etcRoad", 0:"background"}
        else:
            raise AttributeError

    def __len__(self):
        return len(self.samples)

    def make_mask(self, size, label):
        mask = np.zeros([size[0], size[1]], dtype=np.uint8)
        for r in range(len(label["features"])):
            road = label["features"][r]["properties"]
            type_name = road["type_name"]
            if type_name not in self.class_2_id.keys(): continue
            temp = road["road_imcoords"].split(",")
            if len(temp) <= 1: continue
            coords = np.array([int(round(float(c))) for c in temp]).reshape(-1, 2)
            cv2.fillPoly(mask, [coords], self.class_2_id[type_name])
        return mask

    def __getitem__(self, idx):
        with open(self.samples[idx][0], "r") as jfile:
            meta = json.load(jfile)
        img = cv2.imread(self.samples[idx][1])
        if self.rgb: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self.make_mask((img.shape[0], img.shape[1]), meta)
        img = F.to_pil_image(img)
        mask = F.to_pil_image(mask)
        crop_size = int(round(self.patch_size * np.random.uniform(self.preproc_manual["crop_size"][0], self.preproc_manual["crop_size"][1]))) if np.random.uniform() < self.preproc_manual["resize_p"] and self.patch_size < 1024 else self.patch_size
        x = 0 if self.patch_size >= 1024 else np.random.randint(img.width - crop_size)
        y = 0 if self.patch_size >= 1024 else np.random.randint(img.height - crop_size)
        img = F.resized_crop(img, y, x, crop_size, crop_size, self.patch_size)
        mask = F.resized_crop(mask, y, x, crop_size, crop_size, self.patch_size, interpolation=PIL.Image.NEAREST)
        if np.random.uniform() < self.preproc_manual["hflip_p"]:
            img = F.hflip(img)
            mask = F.hflip(mask)
        if np.random.uniform() < self.preproc_manual["vflip_p"]:
            img = F.vflip(img)
            mask = F.vflip(mask)
        if np.random.uniform() < self.preproc_manual["color_p"]:
            img = self.color_jitter(img)
        img = F.to_tensor(img)
        img = self.normalize(img)
        mask = torch.as_tensor(np.array(mask))
        return {"img":img, "mask":mask, "path":self.samples[idx]}

if __name__ == "__main__":
    data = NIADataset("/mnt/data/nia/road", 512, True)
    loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    for i, batch in enumerate(loader):
        di = batch
        for b in range(4):
            print(di["path"][0][b], di["path"][1][b])
            img = data.denormalize(di["img"][b]).numpy().transpose(1, 2, 0)[:, :, ::-1]
            cv2.imshow("img", img)
            cv2.imshow("mask", di["mask"][b].numpy().transpose(0, 1) / 6)
            cv2.waitKey()


'''
if __name__ == "__main__":
    colorbook = {"Mortorway":(51, 51, 255), "Primary":(51, 255, 255), "Secondary":(51, 255, 51), "Tertiary":(255, 255, 51), "Residential":(255, 51, 51), "Unclassified":(255, 51, 255)}
    pads = 3
    thick = 30
    palette = np.ones([len(colorbook)*thick + pads*2 + pads * (len(colorbook) - 1), 250 + pads*2, 3], dtype=np.uint8) * 255
    for i, c in enumerate(colorbook.keys()):
        y = pads + i * (thick + pads)
        cv2.rectangle(palette, (pads, y), (100, y+thick), colorbook[c], -1)
        cv2.putText(palette, c, (100 + pads, y + thick - pads), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 2)

    cv2.imshow("palette", palette)
    cv2.waitKey()

    alpha = 0.2
    data_root = "/mnt/data/nia/road"
    data_img = glob.glob(os.path.join(data_root, "asset", "*.png"))
    data_label = glob.glob(os.path.join(data_root, "label", "*.json"))

    for i in range(len(data_label)):
        filename = os.path.splitext(os.path.basename(data_label[i]))
        imgname = os.path.join(data_root, "asset", filename[0] + ".png")
        img = cv2.imread(imgname)

        with open(data_label[i], "r") as jfile:
            meta = json.load(jfile)

        for r in range(len(meta["features"])):
            road = meta["features"][r]["properties"]
            type_name = road["type_name"]
            if type_name not in colorbook.keys():
                print(type_name)
                continue
            temp = road["road_imcoords"].split(",")
            if len(temp) <= 1:
                continue
            coords = np.array([int(round(float(c))) for c in temp]).reshape(-1, 2)
            canvas = np.zeros_like(img)
            cv2.fillPoly(canvas, [coords], colorbook[type_name])
            canvas = cv2.bitwise_or(canvas, img)
            img = cv2.addWeighted(img, alpha, canvas, 1-alpha, 0)

        save_name = "temp/" + filename[0] + "_overlay.png"
        cv2.imwrite(save_name, img)
        print(f"Saved: {save_name}")
'''