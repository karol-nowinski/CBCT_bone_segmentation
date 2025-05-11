
import torch
import torchio as tio
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import nibabel as nib
import numpy as np


def compute_metrics_per_class(pred, true, num_classes):
    metrics = {}
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    for cls in range(1, num_classes):  # pomijamy klasę 0 (tło)
        print(f"Klasa: {cls}")
        pred_cls = (pred_flat == cls).cpu().numpy()
        true_cls = (true_flat == cls).cpu().numpy()

        precision = precision_score(true_cls, pred_cls, zero_division=0)
        recall = recall_score(true_cls, pred_cls, zero_division=0)
        iou = jaccard_score(true_cls, pred_cls, zero_division=0)
        f1 = f1_score(true_cls, pred_cls, zero_division=0)
        dice = 2 * (precision * recall) / (precision + recall + 1e-6)

        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "dice": dice
        }
    return metrics


if __name__ == "__main__":
    print("--- Uruchomienie skryptu wyliczania metryk ---")


    class_number = 29
    ground_truth_path = "Data\\CleanToothFairy2\\labelsTeethAll\\test\\ToothFairy2F_004.mha"
    predicted_path = "Results\\CleanToothFairy_mergedTeeth_Unet3D96\\ToothFairy2F_004_0000_prediction.mha"

    print("--- Wczytywanie obrazów ---")
    gt = nib.load(ground_truth_path).get_fdata().astype(np.uint8)
    pred = nib.load(predicted_path).get_fdata().astype(np.uint8)


    print("--- Zamiana na tensory ---")
    gt_tensor = torch.tensor(gt, dtype=torch.int64)
    pred_tensor = torch.tensor(pred, dtype=torch.int64)

    print(gt_tensor.is_contiguous())
    print(pred_tensor.is_contiguous())

    print("--- Wyliczanie metryk ---")
    metrics = compute_metrics_per_class(pred_tensor,gt_tensor,class_number)

    print(metrics)


    pass