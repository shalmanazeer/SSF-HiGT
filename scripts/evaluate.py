"""
Evaluate a trained SSF-HiGT checkpoint on a held-out test split and print
accuracy / precision / recall / F1 / FPR plus per-class breakdown, in the
same style as Tables 4-7 of the paper.

Usage:
    python scripts/evaluate.py --config configs/default.yaml \
        --dataset data/cicids2017/clean.csv --checkpoint checkpoints/finetune_final.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FlowDatasetConfig, FlowSequenceDataset, load_and_preprocess
from src.data.graph_builder import build_knn_graph
from src.models.ssf_higt import SSFHiGT
from src.utils.metrics import compute_metrics, per_class_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_preprocess(FlowDatasetConfig(csv_path=args.dataset))
    X_test, y_test = data["test"]
    num_features, num_classes = data["num_features"], data["num_classes"]
    class_names = data["class_names"]

    seq_len = cfg["model"]["transformer"]["max_seq_len"] // 4 or 8
    test_ds = FlowSequenceDataset(X_test, y_test, seq_len=seq_len)
    loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    device_feats = X_test[: min(len(X_test), 500)]
    node_features, adj, edge_attr = build_knn_graph(device_feats, k=cfg["graph"]["knn_k"])
    node_features, adj, edge_attr = node_features.to(device), adj.to(device), edge_attr.to(device)

    model = SSFHiGT(
        num_features=num_features,
        num_classes=num_classes,
        d_model=cfg["model"]["d_model"],
        num_heads=cfg["model"]["transformer"]["num_heads"],
        num_transformer_layers=cfg["model"]["transformer"]["num_layers"],
        num_gat_layers=cfg["model"]["gat"]["num_layers"],
        gat_heads=cfg["model"]["gat"]["num_heads"],
        edge_dim=node_features.shape[-1],
        dropout=cfg["model"]["transformer"]["dropout"],
        max_seq_len=seq_len,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_seq, y in loader:
            x_seq = x_seq.to(device)
            out = model(x_seq, node_features, adj, edge_attr)
            preds = out["logits"].argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    metrics = compute_metrics(y_true, y_pred)
    print("=== Overall metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    print("\n=== Per-class report ===")
    report = per_class_report(y_true, y_pred, class_names[:num_classes])
    for cls, vals in report.items():
        print(f"{cls}: P={vals['precision']:.2f} R={vals['recall']:.2f} F1={vals['f1']:.2f} n={vals['support']}")


if __name__ == "__main__":
    main()
