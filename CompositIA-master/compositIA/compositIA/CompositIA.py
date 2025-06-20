# SPDX-License-Identifier: EUPL-1.2
# Copyright 2024 dp-lab Università della Svizzera Italiana

import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import pydicom
from glob import glob

from utils.windowed_utils import *
from utils.windower import windower
from utils.plots import plotL3
from L3scripts.model_L3 import dice_coef

# ---------------------------------------------------------------------------------------------

def L3segmentation(img, outFolder, model):
    """Segment L3 CT slice into VAT, SAT, and SMA."""
    rgb_img = np.zeros((img.shape[0], img.shape[1], 3))
    rgb_img[:, :, 0] = windower(img, -1024, 2048)
    rgb_img[:, :, 1] = windower(img, -190, -30)
    rgb_img[:, :, 2] = windower(img, 40, 100)

    height, width = img.shape

    X = np.zeros((1, height, width, 3), dtype=np.float32)
    X[0, :, :, :] = rgb_img / 255.

    X = skimage.transform.resize(
        X, (1, height, width, 3),
        mode='constant',
        cval=0,
        anti_aliasing=True,
        preserve_range=True,
        order=3
    )

    results = model.predict(X, verbose=0)

    results = skimage.transform.resize(
        results, (1, height, width, results.shape[-1]),
        mode='constant',
        cval=0,
        anti_aliasing=True,
        preserve_range=True,
        order=0
    )

    res_seg = np.round(results)[0, :, :, :]

    # Zero out edges
    thickness = 20
    res_seg[:, :thickness, 1:] = 0
    res_seg[:, -thickness:, 1:] = 0

    plotL3(img, res_seg, outFolder)

    vat = res_seg[:, :, 1].astype(int)
    sat = res_seg[:, :, 2].astype(int)
    sma = res_seg[:, :, 3].astype(int)

    return vat, sat, sma

# ---------------------------------------------------------------------------------------------

def extract_hounsfield_units(dicom_fp):
    dicom_data = pydicom.dcmread(dicom_fp)
    intercept = dicom_data.get('RescaleIntercept', 0)
    slope = dicom_data.get('RescaleSlope', 1)
    hu_image = dicom_data.pixel_array * slope + intercept
    return hu_image

# ---------------------------------------------------------------------------------------------

def overlay_masks(base_img, VAT_mask, SAT_mask, SMA_mask):
    base_img_norm = (base_img - np.min(base_img)) / (np.max(base_img) - np.min(base_img))
    base_rgb = np.stack([base_img_norm] * 3, axis=-1)

    overlay = base_rgb.copy()
    overlay[VAT_mask == 1] = [1, 0, 0]   # Red for VAT
    overlay[SAT_mask == 1] = [0, 1, 0]   # Green for SAT
    overlay[SMA_mask == 1] = [0, 0, 1]   # Blue for SMA

    return overlay

# ---------------------------------------------------------------------------------------------

def process(MODEL_L3, img_path, outFolder, base_name):
    L3_slice = extract_hounsfield_units(img_path)

    VAT_mask, SAT_mask, SMA_mask = L3segmentation(L3_slice, outFolder, MODEL_L3)

    overlay = overlay_masks(L3_slice, VAT_mask, SAT_mask, SMA_mask)

    os.makedirs(outFolder, exist_ok=True)
    save_path = os.path.join(outFolder, f"{base_name}_compositia.png")
    plt.imsave(save_path, overlay)

# ---------------------------------------------------------------------------------------------

def main(args):
    img_path = args.input_path
    out_path = args.output_folder
    os.makedirs(out_path, exist_ok=True)

    WEIGHTS_L3 = args.weights_L3

    print('Loading model ...')
    MODEL_L3 = tf.keras.models.load_model(WEIGHTS_L3, custom_objects={"dice_coef": dice_coef})

    print('Processing DICOM files...')
    dicom_files = glob(os.path.join(img_path, "*.dcm"))

    for dicom_fp in dicom_files:
        filename = os.path.basename(dicom_fp)
        name_without_ext = os.path.splitext(filename)[0]

        print(f"Processing {filename} ...")
        process(MODEL_L3, dicom_fp, out_path, name_without_ext)

    print("Segmentation completed for all files.")

# ---------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Folder containing .dcm files")
    parser.add_argument("--output_folder", help="Path to output")
    parser.add_argument("--weights_L3", help="Path to L3 segmentation weights", default="./slicer/weights/unet_L3.hdf5")
    args = parser.parse_args()
    main(args)
