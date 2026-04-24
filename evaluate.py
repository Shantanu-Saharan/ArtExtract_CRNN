import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import WikiArtMultiTaskDataset
from models.multitask_crnn import MultiTaskCRNN
from utils.metrics import compute_classification_metrics, top_k_accuracy
from utils.transforms import get_val_transforms


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()

    all_preds = {"artist": [], "style": [], "genre": []}
    all_labels = {"artist": [], "style": [], "genre": []}
    all_logits = {"artist": [], "style": [], "genre": []}

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        
        outputs = model(images)
        
        for task in ["artist", "style", "genre"]:
            task_logits = outputs[task]
            task_preds = torch.argmax(task_logits, dim=1)
            task_labels = labels[task].to(device)

            all_preds[task].extend(task_preds.cpu().numpy().tolist())
            all_labels[task].extend(task_labels.cpu().numpy().tolist())
            all_logits[task].append(task_logits.cpu())

    for task in ["artist", "style", "genre"]:
        all_logits[task] = torch.cat(all_logits[task], dim=0)
    
    return all_labels, all_preds, all_logits


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", square=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_classification_report_image(report_dict, save_path):
    df = pd.DataFrame(report_dict).transpose()
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Classification Report")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MultiTask-CRNN model")
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_artist_classes", type=int, required=True)
    parser.add_argument("--num_style_classes", type=int, required=True)
    parser.add_argument("--num_genre_classes", type=int, required=True)
    parser.add_argument("--artist_classes", type=str, default="")
    parser.add_argument("--style_classes", type=str, default="")
    parser.add_argument("--genre_classes", type=str, default="")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    dataset = WikiArtMultiTaskDataset(
        csv_file=args.val_csv,
        root_dir=args.root_dir,
        transform=get_val_transforms(args.image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = MultiTaskCRNN(
        num_artist_classes=args.num_artist_classes,
        num_style_classes=args.num_style_classes,
        num_genre_classes=args.num_genre_classes,
        backbone=checkpoint.get("backbone", "convnext_base"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))

    class_names = {}
    if args.artist_classes:
        class_names["artist"] = load_class_names(args.artist_classes)
    if args.style_classes:
        class_names["style"] = load_class_names(args.style_classes)
    if args.genre_classes:
        class_names["genre"] = load_class_names(args.genre_classes)

    all_labels, all_preds, all_logits = evaluate_model(model, loader, device)

    metrics_multitask = {}
    num_classes_map = {
        "artist": args.num_artist_classes,
        "style": args.num_style_classes,
        "genre": args.num_genre_classes,
    }

    for task in ["artist", "style", "genre"]:
        y_true = all_labels[task]
        y_pred = all_preds[task]
        logits = all_logits[task]
        
        metrics = compute_classification_metrics(y_true, y_pred)
        label_tensor = torch.tensor(y_true)
        metrics["top5_accuracy"] = top_k_accuracy(logits, label_tensor, k=min(5, num_classes_map[task]))
        
        metrics_multitask[task] = metrics

        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names.get(task) if class_names.get(task) and len(class_names[task]) == num_classes_map[task] else None,
            output_dict=True,
            zero_division=0,
        )

        cm_path = os.path.join(args.output_dir, f"confusion_matrix_{task}.png")
        report_path = os.path.join(args.output_dir, f"classification_report_{task}.png")
        
        save_confusion_matrix(y_true, y_pred, class_names.get(task, []), cm_path)
        save_classification_report_image(report, report_path)

        print(f"\n{task.upper()} Results:")
        print(json.dumps(metrics, indent=2))
        print(f"Saved confusion matrix to {cm_path}")
        print(f"Saved classification report to {report_path}")

    metrics_out_path = os.path.join(args.output_dir, "metrics_multitask.json")
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_multitask, f, indent=2)

    print(f"\nSaved all metrics to {metrics_out_path}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()