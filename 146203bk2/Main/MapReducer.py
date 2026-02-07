import sys
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import Data_Augmentation
import Fusion  # If Fusion.py is in the same directory
from Proposed import DQN

# Define base path
BASE_PATH = r"C:\Bha Project\146203bk2\Main\Processed"


def ensure_directory_exists(directory):
    """Ensure the directory exists before saving files."""
    os.makedirs(directory, exist_ok=True)


def min_max_normalization(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data_minmax = min_max_scaler.fit_transform(data)
    return data_minmax


# Mapper process: split the data to the size of mapper
def mapper(data, target, dts):
    merge_trans = []
    augment_data, augment_label = [], []

    for i in range(len(data)):
        ######### Yeo- Johnson transformation ###########
        data_ = np.array(data[i])  # converting list to array
        power = PowerTransformer(method='yeo-johnson', standardize=True)
        data_trans = power.fit_transform(data_)
        merge_trans.extend(data_trans)

        ######### Feature Fusion (Kulczynki's polynomial) ###########
        sel_feature = Fusion.feature_fusion(data_trans, target[i])

        ######## Augmentation ############################
        aug_data, aug_lab = Data_Augmentation.data_aug(sel_feature, target[i], dts)
        augment_data.append(aug_data)
        augment_label.append(aug_lab)

    # Ensure the directory exists
    ensure_directory_exists(BASE_PATH)

    # Save transformed data
    np.savetxt(os.path.join(BASE_PATH, f"{dts}_trans.csv"), merge_trans, delimiter=',')

    return augment_data, augment_label


# Reducer process
def reducer(fused_data, target, dts, tr_per, acc, sen, spe):
    merge_data, merge_lab = [], []

    for i in range(len(fused_data)):
        x = np.array(fused_data[i])
        y = np.array(target[i])
        merge_data.extend(x)
        merge_lab.extend(y)

    merge_data = np.array(merge_data)
    merge_lab = np.array(merge_lab)

    # Ensure the directory exists
    ensure_directory_exists(BASE_PATH)

    # Save augmented data
    np.savetxt(os.path.join(BASE_PATH, f"{dts}_aug_data.csv"), merge_data, delimiter=',')
    np.savetxt(os.path.join(BASE_PATH, f"{dts}_aug_label.csv"), merge_lab, delimiter=',')

    print("\t\tBig data classification using RFQN.. ")

    # 🔍 Check for reshape compatibility
    if len(merge_data.shape) == 2 and merge_data.shape[1] == 3072:
        try:
            merge_data = merge_data.reshape((merge_data.shape[0], 32, 32, 3))
        except Exception as e:
            print(f"[!] Error while reshaping: {e}")
            return merge_data, merge_lab
    else:
        print(f"[!] Skipping reshape: each sample has {merge_data.shape[1]} features, expected 3072.")

    DQN.classify(merge_data, merge_lab, dts, tr_per, acc, sen, spe)
    return merge_data, merge_lab


# Feature main
def Mapper_phase(data, target, dts):
    feature, cls = mapper(data, target, dts)
    return feature, cls


def Map_Reducer(feat, target, dts, ls, acc, sen, spe):
    print("\t >>Mapper Phase")
    sel_feat, lab = Mapper_phase(feat, target, dts)
    print("\t >>Reducer Phase")
    x, y = reducer(sel_feat, lab, dts, ls, acc, sen, spe)
    return x, y