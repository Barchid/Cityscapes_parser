"""
Parses the CityScapes dataset into a format useful for MDAF-Net, i.e. :
train_rgb, train_depth, train_mask, test_rgb, test_depth, test_mask
"""

import os
import glob
import shutil

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

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def mask_dir_parsing(source_dir, target_dir):
    paths = glob.glob(os.path.join(source_dir, '*', '*labelIds.png')) # take masks
    paths = sorted(paths) # alphanumeric order to keep the same order across source dirs

    for i, path in enumerate(paths):
        new_path = os.path.join(target_dir, str(i) + '.png')
        shutil.copyfile(path, new_path)

def rgb_dir_parsing(source_dir, target_dir):
    paths = glob.glob(os.path.join(source_dir, '*', '*leftImg8bit.png')) # take masks
    paths = sorted(paths) # alphanumeric order to keep the same order across source dirs

    for i, path in enumerate(paths):
        new_path = os.path.join(target_dir, str(i) + '.png')
        shutil.copyfile(path, new_path)

def depth_dir_parsing(source_dir, target_dir):
    paths = glob.glob(os.path.join(source_dir, '*', '*disparity.png')) # take masks
    paths = sorted(paths) # alphanumeric order to keep the same order across source dirs

    for i, path in enumerate(paths):
        new_path = os.path.join(target_dir, str(i) + '.png')
        shutil.copyfile(path, new_path)

# create target folders
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
mask_dir_parsing(SOURCE_VAL_MASK, TARGET_VAL_MASK)
mask_dir_parsing(SOURCE_TEST_MASK, TARGET_TEST_MASK)
mask_dir_parsing(SOURCE_TRAIN_MASK, TARGET_TRAIN_MASK)

depth_dir_parsing(SOURCE_VAL_DEPTH, TARGET_VAL_DEPTH)
depth_dir_parsing(SOURCE_TEST_DEPTH, TARGET_TEST_DEPTH)
depth_dir_parsing(SOURCE_TRAIN_DEPTH, TARGET_TRAIN_DEPTH)

rgb_dir_parsing(SOURCE_VAL_RGB, TARGET_VAL_RGB)
rgb_dir_parsing(SOURCE_TEST_RGB, TARGET_TEST_RGB)
rgb_dir_parsing(SOURCE_TRAIN_RGB, TARGET_TRAIN_RGB)