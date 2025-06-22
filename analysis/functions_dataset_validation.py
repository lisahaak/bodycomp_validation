# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 09:05:27 2025

@author: jjkool
"""

#%% Imports
import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%% Functies

def get_file_paths(base_path, scan_id):
    dicom_fp = os.path.join(base_path, 'original_l3_selected', f'{scan_id}.dcm')
    tag_manual_fp = os.path.join(base_path, 'segmentation_manual_selected', f'{scan_id}.tag')
    png_automatica_fp = os.path.join(base_path, 'segmentation_automatica_selected', f'{scan_id}_automatica.png')
    png_compo_fp = os.path.join(base_path, 'segmentation_compo_selected', f'{scan_id}_compositai.png')
    return dicom_fp, tag_manual_fp, png_automatica_fp, png_compo_fp

def load_manual_masks(tag_fp, image_shape):
    with open(tag_fp, 'rb') as f:
        f.seek(288)
        tags = np.fromfile(f, dtype=np.uint8).reshape(image_shape)
    label_map = {1: "SM", 5: "VAT", 2: "IMAT", 7: "SAT"}
    masks = {label_map.get(lbl): (tags == lbl).astype(np.uint8)
             for lbl in np.unique(tags) if lbl != 0 and label_map.get(lbl) is not None}
    return masks

def load_auto_masks_automatica(png_fp):
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

def load_auto_masks_compo(png_fp):
    auto_img = mpimg.imread(png_fp)
    if auto_img.shape[-1] == 4:
        auto_img = auto_img[..., :3]
    if auto_img.dtype in [np.float32, np.float64]:
        auto_img = (auto_img * 255).round().astype(np.uint8)

    colormap = {
        (255, 0, 0): 'VAT',   # rood
        (0, 255, 0): 'SAT',   # groen
        (0, 0, 255): 'SM',    # blauw
    }

    masks = {}
    for rgb, label in colormap.items():
        mask = np.all(auto_img == rgb, axis=-1).astype(np.uint8)
        if mask.sum() > 0:
            masks[label] = mask
    return masks

def compute_dsc(manual_masks, auto_masks, tissues=['VAT', 'SM', 'SAT']):
    def dsc(tissue):
       
        # alleen weefsels vergelijken die handmatig zijn ingetekend
        if tissue not in manual_masks:
            return np.nan
       
        inter = np.logical_and(manual_masks[tissue], auto_masks[tissue]).sum()
        union = manual_masks[tissue].sum() + auto_masks[tissue].sum()
        return (2 * inter / union) if union > 0 else np.nan
    return {t: dsc(t) for t in tissues}

def plot_segmentation(base_path, scan_id):
    dicom_fp, tag_fp, png_automatica_fp, png_compo_fp = get_file_paths(base_path, scan_id)
    dicom_data = pydicom.dcmread(dicom_fp)
    image = dicom_data.pixel_array

    with open(tag_fp, 'rb') as f:
        f.seek(288)
        tags = np.fromfile(f, dtype=np.uint8).reshape(image.shape)

    auto_img = mpimg.imread(png_compo_fp)
    if auto_img.shape[-1] == 4:
        auto_img = auto_img[..., :3]
    if auto_img.dtype in [np.float32, np.float64]:
        auto_img = (auto_img * 255).round().astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f"Originele scan - {scan_id}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.imshow(tags, cmap='jet', alpha=0.4)
    plt.title(f"Handmatige segmentatie - {scan_id}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.imshow(auto_img)
    plt.title(f"Automatische segmentatie - {scan_id}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def process_scans_dsc(base_path, ground_truth='manual', test_set='automatica'):
    results = []
    original_dir = os.path.join(base_path, 'original_l3_selected')

    scan_ids = [f[:-4] for f in os.listdir(original_dir) if f.endswith('.dcm')]

    for scan_id in scan_ids:
        dicom_fp, tag_fp, png_auto_fp, png_compo_fp = get_file_paths(base_path, scan_id)

        if test_set == 'automatica':
            test_fp = png_auto_fp
            load_auto_masks_fn = load_auto_masks_automatica
        elif test_set == 'compo':
            test_fp = png_compo_fp
            load_auto_masks_fn = load_auto_masks_compo


        if not all([os.path.isfile(f) for f in [dicom_fp, tag_fp, test_fp]]):
            print(f"Mist bestanden voor: {scan_id}")
            continue

        try:
            dicom_data = pydicom.dcmread(dicom_fp)
            img_shape = dicom_data.pixel_array.shape
            manual_masks = load_manual_masks(tag_fp, img_shape)
            auto_masks = load_auto_masks_fn(test_fp)
            dsc_scores = {'scan_id': scan_id}
            dsc_scores.update(compute_dsc(manual_masks, auto_masks))
            results.append(dsc_scores)
            dsc_str = ", ".join(
                f"{t}: {dsc_scores[t]:.4f}" if not np.isnan(dsc_scores[t]) else f"{t}: NaN"
                for t in ['VAT', 'SM', 'SAT']
            )
            print(f"{scan_id} | DSC: {dsc_str}")
        except Exception as e:
            print(f"Fout bij scan {scan_id}: {e}")

    return pd.DataFrame(results)

def plot_statistics_dsc(df, dataset_name):
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
    plt.title(f"DSC per weefseltype ({dataset_name})")
    plt.ylabel("Dice Similarity Coefficient")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def process_scans_area(base_path, ground_truth='manual', test_set='automatica'):
    def compute_area_mm2(masks, spacing_x, spacing_y):
        return {t: mask.sum() * spacing_x * spacing_y for t, mask in masks.items()}

    results = []
    original_dir = os.path.join(base_path, 'original_l3_selected')

    scan_ids = [f[:-4] for f in os.listdir(original_dir) if f.endswith('.dcm')]
    loader = load_auto_masks_automatica if test_set == 'automatica' else load_auto_masks_compo

    for scan_id in scan_ids:
        dicom_fp, tag_fp, png_auto_fp, png_compo_fp = get_file_paths(base_path, scan_id)
        test_fp = png_auto_fp if test_set == 'automatica' else png_compo_fp

        if not all(map(os.path.isfile, [dicom_fp, tag_fp, test_fp])):
            print(f"Mist bestanden voor: {scan_id}")
            continue

        try:
            dicom = pydicom.dcmread(dicom_fp)
            spacing_y, spacing_x = map(float, dicom.PixelSpacing)
            img_shape = dicom.pixel_array.shape

            manual_masks = load_manual_masks(tag_fp, img_shape)
            auto_masks = loader(test_fp)

            manual_areas = compute_area_mm2(manual_masks, spacing_x, spacing_y)
            auto_areas = compute_area_mm2(auto_masks, spacing_x, spacing_y)

            row = {'scan_id': scan_id}
            for t in ['VAT', 'SM', 'SAT']:
                row[f'{t}_manual_mm2'] = manual_areas.get(t, np.nan)
                row[f'{t}_auto_mm2'] = auto_areas.get(t, np.nan)
            results.append(row)

            print(f"{scan_id} | " + ", ".join(f"{t}: {row[f'{t}_manual_mm2']:.1f}/{row[f'{t}_auto_mm2']:.1f}"
                                               if not np.isnan(row[f'{t}_manual_mm2']) else f"{t}: NaN"
                                               for t in ['VAT', 'SM', 'SAT']))
        except Exception as e:
            print(f"Fout bij scan {scan_id}: {e}")

    return pd.DataFrame(results)

def plot_statistics_area(df, dataset_name):
    tissues = ['VAT', 'SM', 'SAT']
    diffs = {}

    print(f"\nArea Differences (Auto - Manual) in mm² for {dataset_name}:\n{'Tissue':<5} | {'Mean':>10} | {'Median':>10} | {'Std Dev':>10} | n")
    print("-" * 60)

    for t in tissues:
        manual = df[f"{t}_manual_mm2"]
        auto = df[f"{t}_auto_mm2"]
        valid = manual.notna() & auto.notna()
        diff = auto[valid] - manual[valid]
        diffs[t] = diff

        print(f"{t:<5} | {diff.mean():10.1f} | {diff.median():10.1f} | {diff.std():10.1f} | {len(diff)}")

    plt.figure(figsize=(8, 6))
    plt.boxplot([diffs[t] for t in tissues], labels=tissues, showmeans=True)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Verschil in oppervlakte per weefseltype ({dataset_name})")
    plt.ylabel("Verschil in oppervlakte (automatisch - handmatig) (mm²)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    for t in tissues:
        manual = df[f"{t}_manual_mm2"]
        auto = df[f"{t}_auto_mm2"]
        valid = manual.notna() & auto.notna()
        x = manual[valid]
        y = auto[valid]

        if len(x) == 0:
            continue

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue', label='Data points')

        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            trendline = np.poly1d(coeffs)
            plt.plot(x, trendline(x), color='green', label=f'Trend line: y={coeffs[0]}x + {coeffs[1]:.2f}')

        plt.xlabel(f"Oppervlakte van handmatige segmentatie voor {t} (mm²)")
        plt.ylabel(f"Verschil in oppervlakte (automatisch - handmatig) voor {t} (mm²)")
        plt.title(f"Verschil in oppervlakte tussen handmatige en automatische segmentatie voor {t}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def process_scans_hu(base_path, ground_truth='manual', test_set='automatica'):
    results = []
    original_dir = os.path.join(base_path, 'original_l3_selected')

    scan_ids = [f[:-4] for f in os.listdir(original_dir) if f.endswith('.dcm')]
    loader = load_auto_masks_automatica if test_set == 'automatica' else load_auto_masks_compo if test_set == 'compo' else None

    for scan_id in scan_ids:
        dicom_fp, tag_fp, png_auto_fp, png_compo_fp = get_file_paths(base_path, scan_id)
        test_fp = png_auto_fp if test_set == 'automatica' else png_compo_fp

        if not all(map(os.path.isfile, [dicom_fp, tag_fp, test_fp])):
            print(f"Mist bestanden voor {scan_id}")
            continue

        try:
            dicom = pydicom.dcmread(dicom_fp)
            hu_values = dicom.pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept

            manual_masks = load_manual_masks(tag_fp, dicom.pixel_array.shape)
            auto_masks = loader(test_fp)

            row = {'scan_id': scan_id}
            for t in ['VAT', 'SM', 'SAT']:
                if t in manual_masks and t in auto_masks:
                    if manual_masks[t].sum() > 0 and auto_masks[t].sum() > 0:
                        hu_manual = hu_values[manual_masks[t] > 0]
                        hu_auto = hu_values[auto_masks[t] > 0]
                        row[f'{t}_manual_hu'] = np.mean(hu_manual)
                        row[f'{t}_auto_hu'] = np.mean(hu_auto)
                    else:
                        row[f'{t}_manual_hu'] = np.nan
                        row[f'{t}_auto_hu'] = np.nan
                else:
                    row[f'{t}_manual_hu'] = np.nan
                    row[f'{t}_auto_hu'] = np.nan

            results.append(row)

            print(f"{scan_id} | " + ", ".join(
                f"{t}: {row[f'{t}_manual_hu']:.2f}/{row[f'{t}_auto_hu']:.2f}"
                if not (np.isnan(row[f'{t}_manual_hu']) or np.isnan(row[f'{t}_auto_hu']))
                else f"{t}: NaN" for t in ['VAT', 'SM', 'SAT'])
            )

        except Exception as e:
            print(f"Fout bij scan {scan_id}: {e}")

    return pd.DataFrame(results)


def plot_statistics_hu(df, dataset_name):
    tissues = ['VAT', 'SM', 'SAT']
    diffs = {}

    print(f"\nHU Differences (Auto - Manual) for {dataset_name}:\n{'Tissue':<5} | {'Mean':>10} | {'Median':>10} | {'Std Dev':>10} | n")
    print("-" * 60)

    for t in tissues:
        manual = df[f"{t}_manual_hu"]
        auto = df[f"{t}_auto_hu"]
        valid = manual.notna() & auto.notna()
        diff = auto[valid] - manual[valid]
        diffs[t] = diff

        print(f"{t:<5} | {diff.mean():10.1f} | {diff.median():10.1f} | {diff.std():10.1f} | {len(diff)}")

    # boxplot van HU verschillen
    plt.figure(figsize=(8, 6))
    plt.boxplot([diffs[t] for t in tissues], labels=tissues, showmeans=True)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"Verschil in gemiddelde HU-waarde per weefseltype ({dataset_name})")
    plt.ylabel("Verschil in gemiddelde HU-waarde (automatisch - handmatig)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # scatter plots per weefsel voor HU
    for t in tissues:
        manual = df[f"{t}_manual_hu"]
        auto = df[f"{t}_auto_hu"]
        valid = manual.notna() & auto.notna()
        x = manual[valid]
        y = auto[valid]

        if len(x) == 0:
            continue

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue', label='Data points')

        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            trendline = np.poly1d(coeffs)
            plt.plot(x, trendline(x), color='green', label=f'Trend line: y={coeffs[0]}x + {coeffs[1]:.2f}')

        plt.xlabel(f"Manual mean HU ({t})")
        plt.ylabel(f"Difference (Auto - Manual) mean HU ({t})")
        plt.title(f"HU Difference Between Auto and Manual Segmentation ({t})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

#%% a
base_path_a = path...

# df_dsc_automatica_a = process_scans_dsc(base_path_a, ground_truth='manual', test_set='automatica')
# plot_statistics_dsc(df_dsc_automatica_a, 'AutoMATiCA (subset A)')
# df_dsc_automatica_a.to_csv(os.path.join(base_path_a, 'dsc_automatica_a_12juni.csv'), index=False)

# df_dsc_compo_a = process_scans_dsc(base_path_a, ground_truth='manual', test_set='compo')
# plot_dsc_statistics(df_dsc_compo_a, 'CompositIA (subset A)')

# df_area_automatica_a = process_scans_area(base_path_a, ground_truth='manual', test_set='automatica')
# plot_statistics_area(df_area_automatica_a, 'AutoMATiCA (subset A)')
# df_area_automatica_a.to_csv(os.path.join(base_path_a, 'area_automatica_a_12juni.csv'), index=False)

# df_area_compo_a = process_scans_area(base_path_a, ground_truth='manual', test_set='compo')
# plot_statistics_area(df_area_automatica_a, 'CompositIA (subset A)')

# df_hu_automatica_a = process_scans_hu(base_path_a, ground_truth='manual', test_set='automatica')
# plot_statistics_hu(df_hu_automatica_a, 'AutoMATiCA (subset A)')
# df_hu_automatica_a.to_csv(os.path.join(base_path_a, 'hu_automatica_a_12juni.csv'), index=False)

# df_hu_compo_a = process_scans_hu(base_path_a, ground_truth='manual', test_set='compo')
# plot_statistics_hu(df_hu_compo_a, 'CompositIA (subset A)')

#%% b1
base_path_b1 = path...

# df_dsc_automatica_b1 = process_scans_dsc(base_path_b1, ground_truth='manual', test_set='automatica')
# plot_statistics_dsc(df_dsc_automatica_b1, 'AutoMATiCA (subset B1)')
# df_dsc_automatica_b1.to_csv(os.path.join(base_path_b1, 'dsc_automatica_b1_12juni.csv'), index=False)

# df_dsc_compo_b1 = process_scans(base_path_b1, ground_truth='manual', test_set='compo')
# plot_dsc_statistics(df_dsc_compo_b1, 'CompositIA (subset B1)')

# df_area_automatica_b1 = process_scans_area(base_path_b1, ground_truth='manual', test_set='automatica')
# plot_statistics_area(df_area_automatica_b1, 'AutoMATiCA (subset B1)')
# df_area_automatica_b1.to_csv(os.path.join(base_path_b1, 'area_automatica_b1_12juni.csv'), index=False)

# df_area_compo_b1 = process_scans_area(base_path_b1, ground_truth='manual', test_set='compo')
# plot_statistics_area(df_area_compo_b1, 'CompositIA (subset B1)')

# df_hu_automatica_b1 = process_scans_hu(base_path_b1, ground_truth='manual', test_set='automatica')
# plot_statistics_hu(df_hu_automatica_b1, 'AutoMATiCA (subset B1)')
# df_hu_automatica_b1.to_csv(os.path.join(base_path_b1, 'hu_automatica_b1_12juni.csv'), index=False)

# df_hu_compo_b1 = process_scans_hu(base_path_b1, ground_truth='manual', test_set='compo')
# plot_statistics_hu(df_hu_compo_b1, 'CompositIA (subset B1)')


#%% b2
base_path_b2 = path...

# df_dsc_automatica_b2 = process_scans_dsc(base_path_b2, ground_truth='manual', test_set='automatica')
# plot_statistics_dsc(df_dsc_automatica_b2, 'AutoMATiCA (subset B2)')
# df_dsc_automatica_b2.to_csv(os.path.join(base_path_b2, 'dsc_automatica_b2_12juni.csv'), index=False)

# df_dsc_compo_b2 = process_scans(base_path_b2, ground_truth='manual', test_set='compo')
# plot_dsc_statistics(df_dsc_compo_b2, 'CompositIA (subset B2)')

# df_area_automatica_b2 = process_scans_area(base_path_b2, ground_truth='manual', test_set='automatica')
# plot_statistics_area(df_area_automatica_b2, 'AutoMATiCA (subset B2)')
# df_area_automatica_b2.to_csv(os.path.join(base_path_b2, 'area_automatica_b2_12juni.csv'), index=False)

# df_area_compo_b2 = process_scans_area(base_path_b2, ground_truth='manual', test_set='compo')
# plot_statistics_area(df_area_compo_b2, 'CompositIA (subset B2)')

# df_hu_automatica_b2 = process_scans_hu(base_path_b2, ground_truth='manual', test_set='automatica')
# plot_statistics_hu(df_hu_automatica_b2, 'AutoMATiCA (subset B2)')
# df_hu_automatica_b2.to_csv(os.path.join(base_path_b2, 'hu_automatica_b2_12juni.csv'), index=False)

# df_hu_compo_b2 = process_scans_hu(base_path_b2, ground_truth='manual', test_set='compo')
# plot_statistics_hu(df_hu_compo_b2, 'CompositIA (subset B2)')
