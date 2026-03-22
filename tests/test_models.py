import torch
from models.backbone import ConvBlock


class TestConvBlock:
    def test_output_shape(self):
        block = ConvBlock(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 288, 512)
        out = block(x)
        assert out.shape == (2, 128, 288, 512)

    def test_preserves_spatial_dims(self):
        block = ConvBlock(in_channels=9, out_channels=64)
        x = torch.randn(1, 9, 288, 512)
        out = block(x)
        assert out.shape[2:] == x.shape[2:]

    def test_groupnorm_num_groups(self):
        block = ConvBlock(in_channels=64, out_channels=128)
        assert isinstance(block.norm, torch.nn.GroupNorm)
        assert block.norm.num_groups == 8

    def test_kaiming_init(self):
        block = ConvBlock(in_channels=64, out_channels=128)
        assert block.conv.weight.abs().sum() > 0

    def test_groupnorm_init(self):
        block = ConvBlock(in_channels=64, out_channels=128)
        assert torch.allclose(block.norm.weight, torch.ones_like(block.norm.weight))
        assert torch.allclose(block.norm.bias, torch.zeros_like(block.norm.bias))
