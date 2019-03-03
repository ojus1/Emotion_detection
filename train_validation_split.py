import shutil
import os
from glob import glob
import random

full_dir = "../Datasets/emotion_detection/full/"

items = glob(full_dir + "*.mp4")

#Generate random file names for validation set
val_set = random.sample(items, int(len(items) * 0.15))

#Copy sampled file names from parent dataset directory to validation folder
for name in val_set:
    shutil.copy(name, "data/video/validation/")
    items.remove(name)

#Copy sampled file names from parent dataset directory to train folder
for name in items:
    shutil.copy(name, "data/video/train/")