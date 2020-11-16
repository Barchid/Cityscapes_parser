"""
Parses the CityScapes dataset into a format useful for MDAF-Net, i.e. :
train_rgb, train_depth, train_mask, test_rgb, test_depth, test_mask
"""

import os
import glob
import shutil
import cv2
import numpy as np

# Source paths from raw dataset
DATASET_PATH = "Cityscapes"
SOURCE_MASK = os.path.join(DATASET_PATH, 'gtFine')
SOURCE_TRAIN_MASK = os.path.join(SOURCE_MASK, 'train')
SOURCE_VAL_MASK = os.path.join(SOURCE_MASK, 'val')
SOURCE_TEST_MASK = os.path.join(SOURCE_MASK, 'test')

SOURCE_RGB = os.path.join(DATASET_PATH, 'leftImg8bit')
SOURCE_TRAIN_RGB = os.path.join(SOURCE_RGB, 'train')
SOURCE_VAL_RGB = os.path.join(SOURCE_RGB, 'val')
SOURCE_TEST_RGB = os.path.join(SOURCE_RGB, 'test')

SOURCE_DEPTH = os.path.join(DATASET_PATH, 'disparity')
SOURCE_TRAIN_DEPTH = os.path.join(SOURCE_DEPTH, 'train')
SOURCE_VAL_DEPTH = os.path.join(SOURCE_DEPTH, 'val')
SOURCE_TEST_DEPTH = os.path.join(SOURCE_DEPTH, 'test')

# Target paths for MDAF-Net
TARGET_PATH = "cs_parsed"
TARGET_TRAIN_MASK = os.path.join(TARGET_PATH, "train_mask")
TARGET_TRAIN_RGB = os.path.join(TARGET_PATH, "train_rgb")
TARGET_TRAIN_DEPTH = os.path.join(TARGET_PATH, "train_depth")
TARGET_VAL_MASK = os.path.join(TARGET_PATH, "val_mask")
TARGET_VAL_RGB = os.path.join(TARGET_PATH, "val_rgb")
TARGET_VAL_DEPTH = os.path.join(TARGET_PATH, "val_depth")
TARGET_TEST_MASK = os.path.join(TARGET_PATH, "test_mask")
TARGET_TEST_RGB = os.path.join(TARGET_PATH, "test_rgb")
TARGET_TEST_DEPTH = os.path.join(TARGET_PATH, "test_depth")

mapping_cs = {
    0: 19,
    1: 19,
    2: 19,
    3: 19,
    4: 19,
    5: 19,
    6: 19,
    7: 0,
    8: 1,
    9: 19,
    10: 19,
    11: 2,
    12: 3,
    13: 4,
    14: 19,
    15: 19,
    16: 19,
    17: 5,
    18: 19,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 19,
    30: 19,
    31: 16,
    32: 17,
    33: 18
}


def convert_mask_ids(dir):
    """
    Converts the mask ids from the 33 IDS setup to a 20 ids setup where 0-118 is the default 19 classes setup and the 19th id is the void id
    (Will be converted to 255 train ID for evaluation on CityScapes)
    """
    for path in os.listdir(dir):
        path = os.path.join(dir, path)
        
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = np.zeros_like(img)

        for oldId, newId in mapping_cs.items():
            mask[img == oldId] = newId

        # replace the image
        cv2.imwrite(path, mask)
        print(path, ' converted successfully')


def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def mask_dir_parsing(source_dir, target_dir):
    paths = glob.glob(os.path.join(
        source_dir, '*', '*labelIds.png'))  # take masks
    # alphanumeric order to keep the same order across source dirs
    paths = sorted(paths)

    for i, path in enumerate(paths):
        new_path = os.path.join(target_dir, str(i) + '.png')
        shutil.copyfile(path, new_path)


def rgb_dir_parsing(source_dir, target_dir):
    paths = glob.glob(os.path.join(
        source_dir, '*', '*leftImg8bit.png'))  # take masks
    # alphanumeric order to keep the same order across source dirs
    paths = sorted(paths)

    for i, path in enumerate(paths):
        new_path = os.path.join(target_dir, str(i) + '.png')
        shutil.copyfile(path, new_path)


def depth_dir_parsing(source_dir, target_dir):
    paths = glob.glob(os.path.join(
        source_dir, '*', '*disparity.png'))  # take masks
    # alphanumeric order to keep the same order across source dirs
    paths = sorted(paths)

    for i, path in enumerate(paths):
        new_path = os.path.join(target_dir, str(i) + '.png')
        shutil.copyfile(path, new_path)


# create target folders
print('TARGET FOLDERS CREATIONS')
mkdir(TARGET_PATH)
mkdir(TARGET_TRAIN_MASK)
mkdir(TARGET_TRAIN_RGB)
mkdir(TARGET_TRAIN_DEPTH)
mkdir(TARGET_VAL_MASK)
mkdir(TARGET_VAL_RGB)
mkdir(TARGET_VAL_DEPTH)
mkdir(TARGET_TEST_MASK)
mkdir(TARGET_TEST_RGB)
mkdir(TARGET_TEST_DEPTH)

# parsing mask folders
print('PARSING MASK DIRECTORIES')
mask_dir_parsing(SOURCE_VAL_MASK, TARGET_VAL_MASK)
mask_dir_parsing(SOURCE_TEST_MASK, TARGET_TEST_MASK)
mask_dir_parsing(SOURCE_TRAIN_MASK, TARGET_TRAIN_MASK)

print('PARSING DEPTH DIRECTORIES')
depth_dir_parsing(SOURCE_VAL_DEPTH, TARGET_VAL_DEPTH)
depth_dir_parsing(SOURCE_TEST_DEPTH, TARGET_TEST_DEPTH)
depth_dir_parsing(SOURCE_TRAIN_DEPTH, TARGET_TRAIN_DEPTH)

print('PARSING RGB DIRECTORIES')
rgb_dir_parsing(SOURCE_VAL_RGB, TARGET_VAL_RGB)
rgb_dir_parsing(SOURCE_TEST_RGB, TARGET_TEST_RGB)
rgb_dir_parsing(SOURCE_TRAIN_RGB, TARGET_TRAIN_RGB)

print('CONVERSION OF MASK IDS')
convert_mask_ids(TARGET_VAL_MASK)
convert_mask_ids(TARGET_TEST_MASK)
convert_mask_ids(TARGET_TRAIN_MASK)