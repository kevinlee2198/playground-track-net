import torch
from models.backbone import ConvBlock, DownBlock, Bottleneck, UpBlock, UNetBackbone
from models.losses import WBCEFocalLoss
from models.tracknet import TrackNet


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


class TestDownBlock:
    def test_output_shapes(self):
        block = DownBlock(in_channels=9, out_channels=64)
        x = torch.randn(2, 9, 288, 512)
        pooled, skip = block(x)
        assert skip.shape == (2, 64, 288, 512)
        assert pooled.shape == (2, 64, 144, 256)

    def test_down2_shapes(self):
        block = DownBlock(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 144, 256)
        pooled, skip = block(x)
        assert skip.shape == (2, 128, 144, 256)
        assert pooled.shape == (2, 128, 72, 128)

    def test_down3_shapes(self):
        block = DownBlock(in_channels=128, out_channels=256)
        x = torch.randn(2, 128, 72, 128)
        pooled, skip = block(x)
        assert skip.shape == (2, 256, 72, 128)
        assert pooled.shape == (2, 256, 36, 64)

    def test_has_two_conv_blocks(self):
        block = DownBlock(in_channels=9, out_channels=64)
        assert isinstance(block.conv1, ConvBlock)
        assert isinstance(block.conv2, ConvBlock)


class TestBottleneck:
    def test_output_shape(self):
        block = Bottleneck(in_channels=256, out_channels=512)
        x = torch.randn(2, 256, 36, 64)
        out = block(x)
        assert out.shape == (2, 512, 36, 64)

    def test_has_three_conv_blocks(self):
        block = Bottleneck(in_channels=256, out_channels=512)
        assert isinstance(block.conv1, ConvBlock)
        assert isinstance(block.conv2, ConvBlock)
        assert isinstance(block.conv3, ConvBlock)


class TestUpBlock:
    def test_up1_shape(self):
        block = UpBlock(in_channels=768, out_channels=256)
        x = torch.randn(2, 512, 36, 64)
        skip = torch.randn(2, 256, 72, 128)
        out = block(x, skip)
        assert out.shape == (2, 256, 72, 128)

    def test_up2_shape(self):
        block = UpBlock(in_channels=384, out_channels=128)
        x = torch.randn(2, 256, 72, 128)
        skip = torch.randn(2, 128, 144, 256)
        out = block(x, skip)
        assert out.shape == (2, 128, 144, 256)

    def test_up3_shape(self):
        block = UpBlock(in_channels=192, out_channels=64)
        x = torch.randn(2, 128, 144, 256)
        skip = torch.randn(2, 64, 288, 512)
        out = block(x, skip)
        assert out.shape == (2, 64, 288, 512)

    def test_has_two_conv_blocks(self):
        block = UpBlock(in_channels=768, out_channels=256)
        assert isinstance(block.conv1, ConvBlock)
        assert isinstance(block.conv2, ConvBlock)


class TestUNetBackbone:
    def test_output_shape(self):
        model = UNetBackbone(in_channels=9, num_classes=3)
        x = torch.randn(2, 9, 288, 512)
        out = model(x)
        assert out.shape == (2, 3, 288, 512)

    def test_output_range_sigmoid(self):
        model = UNetBackbone(in_channels=9, num_classes=3)
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_v5_input_channels(self):
        model = UNetBackbone(in_channels=13, num_classes=3)
        x = torch.randn(1, 13, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)

    def test_encoder_decoder_structure(self):
        model = UNetBackbone(in_channels=9, num_classes=3)
        assert isinstance(model.down1, DownBlock)
        assert isinstance(model.down2, DownBlock)
        assert isinstance(model.down3, DownBlock)
        assert isinstance(model.bottleneck, Bottleneck)
        assert isinstance(model.up1, UpBlock)
        assert isinstance(model.up2, UpBlock)
        assert isinstance(model.up3, UpBlock)

    def test_parameter_count_reasonable(self):
        model = UNetBackbone(in_channels=9, num_classes=3)
        total = sum(p.numel() for p in model.parameters())
        assert 1_000_000 < total < 50_000_000


class TestWBCEFocalLoss:
    def test_returns_scalar(self):
        loss_fn = WBCEFocalLoss()
        pred = torch.sigmoid(torch.randn(2, 3, 288, 512))
        target = torch.zeros(2, 3, 288, 512)
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_prediction_low_loss(self):
        loss_fn = WBCEFocalLoss()
        target = torch.zeros(1, 3, 32, 32)
        target[:, :, 15:17, 15:17] = 1.0
        pred = target.clone().clamp(1e-6, 1 - 1e-6)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01

    def test_bad_prediction_high_loss(self):
        loss_fn = WBCEFocalLoss()
        target = torch.zeros(1, 3, 32, 32)
        target[:, :, 15:17, 15:17] = 1.0
        pred = (1.0 - target).clamp(1e-6, 1 - 1e-6)
        loss = loss_fn(pred, target)
        assert loss.item() > 1.0

    def test_all_zero_target(self):
        loss_fn = WBCEFocalLoss()
        pred = torch.full((1, 3, 32, 32), 0.1)
        target = torch.zeros(1, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_continuous_targets_mixup(self):
        loss_fn = WBCEFocalLoss()
        pred = torch.sigmoid(torch.randn(1, 3, 32, 32))
        target = torch.rand(1, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flows(self):
        loss_fn = WBCEFocalLoss()
        raw = torch.randn(1, 3, 32, 32, requires_grad=True)
        pred = torch.sigmoid(raw)
        target = torch.zeros(1, 3, 32, 32)
        target[:, :, 15:17, 15:17] = 1.0
        loss = loss_fn(pred, target)
        loss.backward()
        assert raw.grad is not None
        assert raw.grad.abs().sum() > 0


class TestTrackNet:
    def test_v2_forward(self):
        model = TrackNet()
        x = torch.randn(2, 9, 288, 512)
        out = model(x)
        assert out.shape == (2, 3, 288, 512)

    def test_output_range(self):
        model = TrackNet()
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_mdd_slot_none(self):
        model = TrackNet()
        assert model.mdd is None
        assert model.rstr is None

    def test_backbone_accessible(self):
        model = TrackNet()
        assert isinstance(model.backbone, UNetBackbone)
