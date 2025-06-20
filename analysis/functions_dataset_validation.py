# -*- coding: utf-8 -*-
"""
Created on Fri May 23 16:39:10 2025
@author: jjkool
"""

#%% Imports

import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%% Functions

def get_chunk_path(base_path, chunk):
    return os.path.join(base_path, 'original_l3_split', chunk)

def get_file_paths(base_path, scan_id, chunk):
    dicom_fp = os.path.join(base_path, 'original_l3_split', chunk, f'{scan_id}.dcm')
    tag_fp = os.path.join(base_path, 'tagged_manual_split', chunk, f'{scan_id}.tag')
    png_fp = os.path.join(base_path, 'tagged_automatica_split_renamed', chunk, f'{scan_id}_automatica.png')
    return dicom_fp, tag_fp, png_fp

def load_manual_masks(tag_fp, image_shape):
    with open(tag_fp, 'rb') as f:
        f.seek(288)
        tags = np.fromfile(f, dtype=np.uint8).reshape(image_shape)
    label_map = {1: "SM", 5: "VAT", 2: "IMAT", 7: "SAT"}
    masks = {label_map.get(lbl): (tags == lbl).astype(np.uint8) 
             for lbl in np.unique(tags) if lbl != 0 and label_map.get(lbl) is not None}
    return masks

def load_auto_masks(png_fp):
    auto_img = mpimg.imread(png_fp)
    if auto_img.shape[-1] == 4:
        auto_img = auto_img[..., :3]
    if auto_img.dtype in [np.float32, np.float64]:
        auto_img = (auto_img * 255).round().astype(np.uint8)

    pixels = auto_img.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    class_labels = ['IMAT', 'SAT', 'SM', 'VAT']

    masks = {}
    for idx, color in enumerate(unique_colors[1:], start=1):
        if idx > len(class_labels):
            break
        masks[class_labels[idx - 1]] = np.all(auto_img == color, axis=-1).astype(np.uint8)
    return masks

def compute_dsc(manual_masks, auto_masks, tissues=['VAT', 'SM', 'SAT']):
    def dsc(tissue):
        if tissue not in manual_masks or tissue not in auto_masks:
            return np.nan
        inter = np.logical_and(manual_masks[tissue], auto_masks[tissue]).sum()
        denom = manual_masks[tissue].sum() + auto_masks[tissue].sum()
        return (2 * inter / denom) if denom > 0 else np.nan
    return {t: dsc(t) for t in tissues}

def get_dsc_for_scan(base_path, scan_id, chunk):
    dicom_fp, tag_fp, png_fp = get_file_paths(base_path, scan_id, chunk)
    dicom_data = pydicom.dcmread(dicom_fp)
    img_shape = dicom_data.pixel_array.shape

    manual_masks = load_manual_masks(tag_fp, img_shape)
    auto_masks = load_auto_masks(png_fp)

    return compute_dsc(manual_masks, auto_masks)

def plot_segmentation(base_path, scan_id, chunk):
    dicom_fp, tag_fp, png_fp = get_file_paths(base_path, scan_id, chunk)
    dicom_data = pydicom.dcmread(dicom_fp)
    image = dicom_data.pixel_array

    with open(tag_fp, 'rb') as f:
        f.seek(288)
        tags = np.fromfile(f, dtype=np.uint8).reshape(image.shape)

    auto_img = mpimg.imread(png_fp)
    if auto_img.shape[-1] == 4:
        auto_img = auto_img[..., :3]
    if auto_img.dtype in [np.float32, np.float64]:
        auto_img = (auto_img * 255).round().astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f"CT-scan L3 - {scan_id}, {chunk}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.imshow(tags, cmap='jet', alpha=0.4)
    plt.title(f"Manual segmentation - {scan_id}, {chunk}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')    
    plt.imshow(auto_img)
    plt.title(f"Automatic segmentation - {scan_id}, {chunk}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def process_chunks(base_path, num_chunks):
    results = []
    for i in range(1, num_chunks + 1):
        chunk = f'chunk_{i:03d}'
        original_dir = get_chunk_path(base_path, chunk)
        if not os.path.isdir(original_dir):
            print(f"Directory does not exist: {original_dir}")
            continue

        scans = [f[:-4] for f in os.listdir(original_dir) if f.endswith('.dcm')]

        for scan_id in scans:
            dicom_fp, tag_fp, png_fp = get_file_paths(base_path, scan_id, chunk)
            if not (os.path.isfile(dicom_fp) and os.path.isfile(tag_fp) and os.path.isfile(png_fp)):
                print(f"Missing files for scan {scan_id} in {chunk}")
                continue

            try:
                dsc_scores = get_dsc_for_scan(base_path, scan_id, chunk)
                dsc_scores.update({'scan_id': scan_id, 'chunk': chunk})
                results.append(dsc_scores)
                dsc_str = ", ".join([f"{t}: {dsc_scores[t]:.4f}" if not np.isnan(dsc_scores[t]) else f"{t}: NaN" for t in ['VAT', 'SM', 'SAT']])
                print(f"{scan_id}, {chunk} | DSC: {dsc_str}")
            except Exception as e:
                print(f"Error processing scan {scan_id} in {chunk}: {e}")

    return pd.DataFrame(results)

def plot_dsc_statistics(df, dataset_name):
    tissues = ['VAT', 'SM', 'SAT']
    counts = [df[t].dropna().shape[0] for t in tissues]
    labels = [f"{t} (n={c})" for t, c in zip(tissues, counts)]

    print(f"\nDSC Statistics for {dataset_name}:")
    for t in tissues:
        data = df[t].dropna()
        print(f"{t}: Median={np.median(data):.4f}, Mean={np.mean(data):.4f}, Std={np.std(data):.4f}, IQR={np.percentile(data, 75)-np.percentile(data, 25):.4f}")

    plt.figure(figsize=(8, 6))
    plt.boxplot([df[t].dropna() for t in tissues], tick_labels=labels, showmeans=True)
    plt.ylim(0, 1)
    plt.title(f"DSC Distribution per Tissue: {dataset_name}")
    plt.ylabel("Dice Similarity Coefficient")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

#%% Dataset A

base_path_a = path...
df_results_a = process_chunks(base_path_a, 12)
plot_dsc_statistics(df_results_a, 'dataset A')


#%% Dataset B1

base_path_b_b1 = path...

df_results_b_b1 = process_chunks(base_path_b_b1, 1)
plot_dsc_statistics(df_results_b_b1, 'dataset B1')


#%% Dataset B2

base_path_b_b2 = path...

df_results_b_b2 = process_chunks(base_path_b_b2, 1)
plot_dsc_statistics(df_results_b_b2, 'dataset B2')


