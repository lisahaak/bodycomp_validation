import os
import shutil
import numpy as np
import pydicom
import csv

ori_dir = 
tagged_dir = 

output_base =
subsets = ['b0', 'b1', 'b2']

for subset in subsets:
    os.makedirs(os.path.join(output_base, subset, 'original_l3'), exist_ok=True)
    os.makedirs(os.path.join(output_base, subset, 'tagged_manual'), exist_ok=True)

ori_files = [f for f in os.listdir(ori_dir) if not f.endswith('.tag')]
tagged_files = [f for f in os.listdir(tagged_dir) if f.endswith('.tag')]

ori_bases = {os.path.splitext(f)[0]: f for f in ori_files}
tagged_bases = {os.path.splitext(f)[0]: f for f in tagged_files}
common_bases = list(set(ori_bases) & set(tagged_bases))

vat_info = {}

for base in common_bases:
    dicom_path = os.path.join(ori_dir, ori_bases[base])
    tag_path = os.path.join(tagged_dir, tagged_bases[base])

    try:
        # Read DICOM and get pixel spacing
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array
        pixel_spacing = dicom_data.PixelSpacing
        spacing_x = float(pixel_spacing[1])
        spacing_y = float(pixel_spacing[0])

        with open(tag_path, 'rb') as f:
            f.seek(288)
            tags = np.fromfile(f, dtype=np.uint8)
        tags = tags.reshape(image.shape)

        vat_mask = (tags == 5).astype(np.uint8)
        pixel_count = vat_mask.sum()
        vat_area_mm2 = pixel_count * spacing_x * spacing_y

        if vat_area_mm2 == 0:
            subset = 'b0'
        elif vat_area_mm2 < 10000:
            subset = 'b1'
        else:
            subset = 'b2'

        shutil.copy2(dicom_path, os.path.join(output_base, subset, 'original_l3', ori_bases[base]))
        shutil.copy2(tag_path, os.path.join(output_base, subset, 'tagged_manual', tagged_bases[base]))

        vat_info[base] = {'VAT_area_mm2': vat_area_mm2, 'subset': subset}
        print(f"{base}: {vat_area_mm2:.2f} mm² → {subset}")

    except Exception as e:
        print(f" Error with scan {base}: {e}")

csv_path = os.path.join(output_base, 'vat_areas_and_subsets.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['scan_id', 'VAT_area_mm2', 'subset'])
    for base, info in vat_info.items():
        writer.writerow([base, f"{info['VAT_area_mm2']:.2f}", info['subset']])

print(f"All done. Summary saved to {csv_path}")
