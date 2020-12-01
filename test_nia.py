import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from networks.dinknet import LinkNet34, DinkNet34
from networks.nllinknet_location import NL3_LinkNet, NL4_LinkNet, NL34_LinkNet, Baseline
from networks.nllinknet_pairwise_func import NL_LinkNet_DotProduct, NL_LinkNet_Gaussian, NL_LinkNet_EGaussian
from networks.unet import Unet
from test_framework import TTAFramework
from nia_data import NIADataset


def test_models(model, n_class, name, source='../dataset/Road/valid', scales=(1.0,), target=''):
    if type(scales) == tuple:
        scales = list(scales)
    print(model, name, source, scales, target)

    solver = TTAFramework(model, n_class)
    solver.load('weights/' + name + '.th')

    if target == '':
        target = 'submits/' + name + '/'
    else:
        target = 'submits/' + target + '/'
    os.makedirs(target, exist_ok=True)

    val = NIADataset(source, patch_size=1024, shuffle=True, rgb=True, n_class=n_class)

    # val = os.listdir(source)
    # if not os.path.exists(target):
    #     try:
    #         os.makedirs(target)
    #     except OSError as e:
    #         import errno
    #         if e.errno != errno.EEXIST:
    #             raise
    len_scales = int(len(scales))
    if len_scales > 1:
        print('multi-scaled test : ', scales)

    for i, name in tqdm(enumerate(val), ncols=10, desc="Testing "):
        mask = solver.test_one_img_from_path(name["path"][1], scales)
        # mask[mask > 4.0 * len_scales] = 255  # 4.0
        # mask[mask <= 4.0 * len_scales] = 0
        # mask = mask[:, :, None]
        mask = np.argmax(mask, axis=0)
        # mask = np.concatenate([mask, mask, mask], axis=2)

        # TODO (Junghoon): Index - Color Mapping Should be HERE.
        img = cv2.imread(name["path"][1])
        mask_im = np.zeros_like(img)
        alpha = 0.2
        for c in val.class_2_id.keys():
            if c == "background": continue
            c_id = val.class_2_id[c]
            color = val.colorbook[c]
            temp = np.zeros_like(img)
            temp[mask == c_id] = color
            mask_im[mask == c_id] = color
            temp = cv2.bitwise_or(temp, img)
            img = cv2.addWeighted(img, alpha, temp, 1-alpha, 0)
        cv2.imwrite(target + os.path.basename(name["path"][1][:-4]) + '_blend.png', img)
        cv2.imwrite(target + os.path.basename(name["path"][1][:-4]) + '_mask.png', mask_im)
        cv2.imwrite(target + os.path.basename(name["path"][1][:-4]) + '_pred_idx.png', mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="set model name")
    parser.add_argument("--n_class", help="Number of class output", type=int)
    parser.add_argument("--name", help="set path of weights")
    parser.add_argument("--source", help="path of test datasets", default='dataset/valid')
    parser.add_argument("--scales", help="set scales for MST", default=[1.0], type=float, nargs='*')
    parser.add_argument("--target", help="path of submit files", default='')

    args = parser.parse_args()

    models = {'NL3_LinkNet': NL3_LinkNet, 'NL4_LinkNet': NL4_LinkNet, 'NL34_LinkNet': NL34_LinkNet,
              'Baseline': Baseline,
              'NL_LinkNet_DotProduct': NL_LinkNet_DotProduct, 'NL_LinkNet_Gaussian': NL_LinkNet_Gaussian,
              'NL_LinkNet_EGaussian': NL_LinkNet_EGaussian,
              'UNet': Unet, 'LinkNet': LinkNet34, 'DLinkNet': DinkNet34}

    model = models[args.model]
    name = args.name
    n_class = args.n_class
    scales = args.scales
    target = args.target
    source = args.source

    test_models(model=model, n_class=n_class, name=name, source=source, scales=scales, target=target)


if __name__ == "__main__":
    main()
