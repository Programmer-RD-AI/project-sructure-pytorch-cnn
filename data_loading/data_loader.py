import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2
import numpy as np
import os
from data_loading.transforming import transform_data


def load_data(img_size=112):
    data = []
    index = -1
    labels = {}
    for folder in os.listdir("./data/"):
        index += 1
        labels["./data/" + folder + "/"] = index
    for label in labels:
        for file in os.listdir(label):
            img = cv2.imread(label + file)
            img = cv2.resize(img, (img_size, img_size))
            data.append([np.array(transform_data(img)), labels[label]])
    np.random.shuffle(data)
    np.save("./out/cleaned_data/data.npy", data)
    return data
