# Plik zawiera przykładowe ładowanie pliku .nii.gz

from pathlib import Path
import nibabel as nib
import SimpleITK as sitk

folder = Path("Data/ChinaCBCT/label")
# Wyszukaj pierwszy plik .nii lub .nii.gz
nii_files = list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz"))
invalid_files = []


if not nii_files:
    print("❌ Nie znaleziono żadnych plików .nii lub .nii.gz w folderze.")
else:
    for file in nii_files:

        print(f"🔍 Próba załadowania pliku: {file.name}")
        try:
        # Wczytaj obraz
            img = sitk.ReadImage(str(file))

            size = img.GetSize()
            spacing = img.GetSpacing()
            dtype = img.GetPixelIDTypeAsString()
            array = sitk.GetArrayViewFromImage(img)

            value_min = array.min()
            value_max = array.max()

            print(f"\n✅ {file.name}")
            print("  📐 Shape:   ", size)
            print("  📏 Spacing: ", spacing)
            print("  🔄 Dtype:   ", dtype)
            print(f"  🧠 Value range: {value_min} to {value_max}")

             # Opcjonalne sprawdzenie "pustych" plików
            if array.size == 0 or value_max == value_min:
                print("  ⚠️ Uwaga: Podejrzanie jednolite lub puste dane")
                invalid_files.append((file.name, "Jednolite dane"))
        except Exception as e:
            print(f"\n❌ Błąd w pliku {file.name}: {e}")
            invalid_files.append((file.name, str(e)))