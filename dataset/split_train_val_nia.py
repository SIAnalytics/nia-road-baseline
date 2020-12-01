import os
import argparse
import shutil
from random import shuffle


def get_filenames(label_dataset_path):
    filenames = []
    for filename in os.listdir(label_dataset_path):
        if filename.endswith(".json"):
            filenames.append(filename[:-5])
    return filenames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="the path of datasets", default="/nas/datasets/RSI_OP_NIA_PUB3/road")
    parser.add_argument("--val_frac", help="Fraction of validation datasets. 0 < val_frac < 1.", type=float, default=0.1)
    parser.add_argument("--seed", help="Random seed", type=int, default=2020)
    args = parser.parse_args()

    root = args.dataset
    label_root = os.path.join(root, 'label')
    asset_root = os.path.join(root, 'asset')

    filenames = get_filenames(label_root)
    shuffle(filenames)
    
    train_frac = 1 - args.val_frac
    train_num = int(len(filenames) * train_frac)
    filenames_train = filenames[:train_num]
    filenames_val = filenames[train_num:]

    for filename_set, set_name in ((filenames_train, 'train'), ((filenames_val), 'valid')):
        shutil.rmtree(set_name)
        os.makedirs(os.path.join(set_name, 'asset'), exist_ok=True)
        os.makedirs(os.path.join(set_name, 'label'), exist_ok=True)

        for filename_each in filename_set:
            filename_png = filename_each + '.png'
            filename_kml = filename_each + '.kml'
            filename_tif = filename_each + '.tif'
            filename_json = filename_each + '.json'

            old_path_png = os.path.join(asset_root, filename_png)
            old_path_kml = os.path.join(asset_root, filename_kml)
            old_path_tif = os.path.join(asset_root, filename_tif)
            old_path_json = os.path.join(label_root, filename_json)

            new_path_png = os.path.join(set_name, 'asset', filename_png)
            new_path_kml = os.path.join(set_name, 'asset', filename_kml)
            new_path_tif = os.path.join(set_name, 'asset', filename_tif)
            new_path_json = os.path.join(set_name, 'label', filename_json)

            os.symlink(old_path_png, new_path_png)
            os.symlink(old_path_kml, new_path_kml)
            os.symlink(old_path_tif, new_path_tif)
            os.symlink(old_path_json, new_path_json)


if __name__ == '__main__':
    main()
