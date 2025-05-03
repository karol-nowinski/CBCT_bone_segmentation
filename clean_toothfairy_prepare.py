import SimpleITK as sitk
import numpy as np
import os


# Scalanie zebow do jednej klasy
# def prepare_tooth_mask(input_folder, output_folder):
#     os.makedirs(output_folder, exist_ok=True)

#     for filename in os.listdir(input_folder):
#         if filename.endswith(".mha"):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, filename)

#             img = sitk.ReadImage(input_path)
#             data = sitk.GetArrayFromImage(img)

#             # W przypadku gdy wartość większa od 
#             new_data = np.where(data >=8,8,data).astype(np.uint8)

#             binary_img = sitk.GetImageFromArray(new_data)
#             binary_img.CopyInformation(img)

#             sitk.WriteImage(binary_img, output_path,useCompression=True)

#             print(f"Przetworzono: {filename}")

def prepare_remapped_tooth_mask(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Lista oryginalnych etykiet zębów w kolejności rosnącej
    original_tooth_labels = (
        [8,9,10] +
        list(range(11, 19)) +  # Górne prawe
        list(range(21, 29)) +  # Górne lewe
        list(range(31, 39)) +  # Dolne lewe
        list(range(41, 49))    # Dolne prawe
    )

    # Mapa: oryginalna etykieta -> nowa etykieta (1..N)
    remap_dict = {old_label: new_label for new_label, old_label in enumerate(original_tooth_labels, start=1)}

    for filename in os.listdir(input_folder):
        if filename.endswith(".mha"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = sitk.ReadImage(input_path)
            data = sitk.GetArrayFromImage(img)

            # Nowa maska z wartościami 0 (tło)
            new_data = np.zeros_like(data, dtype=np.uint8)

            # Przepisz etykiety zgodnie z remap_dict
            for old_label, new_label in remap_dict.items():
                new_data[data == old_label] = new_label

            remapped_img = sitk.GetImageFromArray(new_data)
            remapped_img.CopyInformation(img)

            sitk.WriteImage(remapped_img, output_path, useCompression=True)

            print(f"Przetworzono: {filename}")



# Ścieżki
input_folder = "Data\\CleanToothFairy2\\labelsTr\\validation"
output_folder = "Data\\CleanToothFairy2\\labelsOnlyTeeth\\validation"

prepare_remapped_tooth_mask(input_folder,output_folder)

