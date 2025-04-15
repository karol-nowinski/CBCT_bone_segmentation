import os
import nibabel as nib
import numpy as np
from collections import defaultdict

# Ścieżka do folderu z plikami .nii.gz
folder_path = "Data/ChinaCBCT/label"
output_file = "statystyki_klas.txt"

# Zbiorcze zliczanie
total_counts = defaultdict(int)
global_max_class = 0

# Bufor tekstowy na wyniki
lines = []

# Przetwarzanie każdego pliku .nii.gz w folderze
for filename in os.listdir(folder_path):
    if filename.endswith(".nii.gz"):
        file_path = os.path.join(folder_path, filename)

        # Wczytanie danych
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.int32)

        # Zliczanie klas
        unique, counts = np.unique(data, return_counts=True)
        class_counts = dict(zip(unique, counts))

        # Aktualizacja zbiorczych danych
        for label, count in class_counts.items():
            total_counts[int(label)] += int(count)

        # Największy numer klasy w pliku
        file_max = unique.max()
        if file_max > global_max_class:
            global_max_class = int(file_max)

        # Zapis wyników dla pliku
        lines.append(f"\nPlik: {filename}")
        lines.append("Liczność klas:")
        for label, count in class_counts.items():
            lines.append(f"  Klasa {int(label)}: {count} voxeli")
        lines.append(f"Największy numer klasy w pliku: {int(file_max)}")

# Podsumowanie ogólne
lines.append("\n========== PODSUMOWANIE GLOBALNE ==========")
lines.append("Łączna liczność klas we wszystkich plikach:")
for label in sorted(total_counts.keys()):
    lines.append(f"  Klasa {label}: {total_counts[label]} voxeli")
lines.append(f"Największy numer klasy ogółem: {global_max_class}")

# Zapis do pliku
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\n✅ Wyniki zapisane do pliku: {output_file}")
