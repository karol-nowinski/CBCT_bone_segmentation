import nibabel as nib

try:
    img = nib.load("Data/ChinaCBCT/label/1000889125_20171009.nii")
    print("✔️ Działa jako .nii (bez kompresji)")
except Exception as e:
    print("❌ Nadal nie działa:", e)
    