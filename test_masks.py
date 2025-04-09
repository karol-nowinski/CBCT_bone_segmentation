import nibabel as nib
import numpy as np

img = nib.load("D:\Magisterka\mgrRepo\Data\ChinaCBCT\label\X2313838.nii.gz")
data = img.get_fdata()

labels = np.unique(data)
print("Występujące klasy:", labels)