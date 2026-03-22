import yaml
import os

def test_config_loads_from_yaml():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert config["optimizer"] == "AdamW"
    assert config["learning_rate"] == 1e-4
    assert config["lr_schedule"]["name"] == "MultiStepLR"
    assert config["lr_schedule"]["milestones"] == [20, 25]
    assert config["lr_schedule"]["gamma"] == 0.1
    assert config["batch_size"] == 2
    assert config["epochs"] == 30
    assert config["input_size"] == [512, 288]
    assert config["seq_len"] == 3
    assert config["heatmap_radius"] == 30
    assert config["num_workers"] == 4
    assert config["pin_memory"] is True
    assert config["persistent_workers"] is True
    assert config["amp_dtype"] == "bfloat16"
    assert config["compile_model"] is True
    assert config["seed"] == 42
    assert config["checkpoint_dir"] == "checkpoints"
    assert config["log_dir"] == "runs"


import torch
from training.evaluate import heatmap_to_position, compute_detection_metrics, aggregate_metrics

def test_heatmap_to_position_with_ball():
    heatmap = torch.zeros(288, 512)
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            if dx * dx + dy * dy <= 25:
                heatmap[150 + dy, 100 + dx] = 0.9
    x, y, detected = heatmap_to_position(heatmap, threshold=0.5)
    assert detected is True
    assert abs(x - 100) < 2
    assert abs(y - 150) < 2

def test_heatmap_to_position_no_ball():
    heatmap = torch.zeros(288, 512)
    x, y, detected = heatmap_to_position(heatmap, threshold=0.5)
    assert detected is False

def test_compute_detection_metrics_true_positive():
    pred = (100.0, 150.0, True)
    gt = (102.0, 151.0, True)
    tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
    assert tp == 1 and fp == 0 and fn == 0

def test_compute_detection_metrics_false_positive():
    pred = (100.0, 150.0, True)
    gt = (200.0, 200.0, True)
    tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
    assert tp == 0 and fp == 1 and fn == 1

def test_compute_detection_metrics_false_negative():
    pred = (0.0, 0.0, False)
    gt = (100.0, 150.0, True)
    tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
    assert tp == 0 and fp == 0 and fn == 1

def test_compute_detection_metrics_true_negative():
    pred = (0.0, 0.0, False)
    gt = (0.0, 0.0, False)
    tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
    assert tp == 0 and fp == 0 and fn == 0

def test_compute_detection_metrics_fp_no_gt():
    pred = (100.0, 150.0, True)
    gt = (0.0, 0.0, False)
    tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
    assert tp == 0 and fp == 1 and fn == 0

def test_aggregate_metrics():
    metrics = aggregate_metrics(tp=7, fp=2, fn=1)
    assert abs(metrics["precision"] - 7 / 9) < 1e-6
    assert abs(metrics["recall"] - 7 / 8) < 1e-6
    expected_f1 = 2 * (7 / 9) * (7 / 8) / ((7 / 9) + (7 / 8))
    assert abs(metrics["f1"] - expected_f1) < 1e-6

def test_aggregate_metrics_zero_division():
    metrics = aggregate_metrics(tp=0, fp=0, fn=0)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from training.evaluate import evaluate_epoch

def _make_synthetic_heatmap(x, y, h=288, w=512, radius=5):
    heatmap = torch.zeros(h, w)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                py, px = y + dy, x + dx
                if 0 <= py < h and 0 <= px < w:
                    heatmap[py, px] = 1.0
    return heatmap

class PerfectModel(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self._outputs = outputs
        self._call_idx = 0
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self._outputs[self._call_idx]
        self._call_idx += 1
        return out

def test_evaluate_epoch_all_tp():
    gt1 = torch.stack([
        _make_synthetic_heatmap(100, 150),
        _make_synthetic_heatmap(200, 100),
        _make_synthetic_heatmap(300, 200),
    ])
    gt2 = torch.stack([
        _make_synthetic_heatmap(50, 50),
        _make_synthetic_heatmap(400, 250),
        _make_synthetic_heatmap(250, 144),
    ])
    frames = torch.randn(2, 9, 288, 512)
    gt_heatmaps = torch.stack([gt1, gt2])
    dataset = TensorDataset(frames, gt_heatmaps)
    dataloader = DataLoader(dataset, batch_size=2)
    model = PerfectModel(outputs=[gt_heatmaps])
    device = torch.device("cpu")
    metrics = evaluate_epoch(model, dataloader, device)
    assert metrics["tp"] == 6
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["f1"] == 1.0
