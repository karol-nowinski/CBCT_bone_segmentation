import nibabel as nib
import numpy as np
import os

# Skrypt przygotowujący clean dataset- wszytskie zęby tworza jedną klasę

def prepare_label(input_folder,output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".nii.gz"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = nib.load(input_path)
            data = img.get_fdata()

            binary_data = (data > 0).astype(np.uint8)

            binary_img = nib.Nifti1Image(binary_data, img.affine, img.header)
            nib.save(binary_img, output_path)

            print(f"Przetworzono: {filename}")



input_folder = "Data\ChinaCBCT\label"
output_folder = "Data\ChinaCBCTClean\label"

prepare_label(input_folder,output_folder)