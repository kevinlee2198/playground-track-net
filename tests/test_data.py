import torch
import pytest
from data.heatmap import generate_heatmap


class TestGenerateHeatmap:
    def test_visible_ball_returns_circle(self):
        heatmap = generate_heatmap(x=256, y=144, visibility=1, height=288, width=512, radius=30)
        assert heatmap.shape == (288, 512)
        assert heatmap.dtype == torch.float32
        assert heatmap[144, 256] == 1.0
        assert heatmap[0, 0] == 0.0

    def test_invisible_ball_returns_zeros(self):
        heatmap = generate_heatmap(x=0, y=0, visibility=0, height=288, width=512, radius=30)
        assert heatmap.shape == (288, 512)
        assert heatmap.sum() == 0.0

    def test_partially_occluded_still_labeled(self):
        heatmap = generate_heatmap(x=100, y=100, visibility=2, height=288, width=512, radius=30)
        assert heatmap[100, 100] == 1.0
        assert heatmap.sum() > 0.0

    def test_circle_radius(self):
        heatmap = generate_heatmap(x=256, y=144, visibility=1, height=288, width=512, radius=30)
        assert heatmap[144, 256 + 30] == 1.0
        assert heatmap[144, 256 + 31] == 0.0

    def test_ball_at_edge_clips_to_image(self):
        heatmap = generate_heatmap(x=5, y=5, visibility=1, height=288, width=512, radius=30)
        assert heatmap.shape == (288, 512)
        assert heatmap[5, 5] == 1.0
        centered = generate_heatmap(x=256, y=144, visibility=1, height=288, width=512, radius=30)
        assert heatmap.sum() < centered.sum()

    def test_values_are_binary(self):
        heatmap = generate_heatmap(x=200, y=100, visibility=1, height=288, width=512, radius=30)
        unique_vals = torch.unique(heatmap)
        assert all(v in (0.0, 1.0) for v in unique_vals)
