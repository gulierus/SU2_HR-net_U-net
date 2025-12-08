import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Basic convolutional block."""
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class AttentionGate(nn.Module):
    """Attention Gate for focusing on relevant features."""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNetPlusPlus(nn.Module):
    """U-Net++ with optional attention gates."""
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512], use_attention=True, dropout_rate=0.1):
        super(UNetPlusPlus, self).__init__()
        self.use_attention = use_attention
        self.features = features
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i, feature in enumerate(features):
            in_ch = in_channels if i == 0 else features[i-1]
            self.encoders.append(ConvBlock(in_ch, feature, dropout_rate))
            if i < len(features) - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.ups = nn.ModuleList()
        for i in range(len(features) - 1):
            self.ups.append(nn.ConvTranspose2d(features[i + 1], features[i], kernel_size=2, stride=2))
        self.decoder_convs = nn.ModuleDict()
        for i in range(len(features) - 1):
            self.decoder_convs[f"conv_0_{i}"] = ConvBlock(features[i] * 2, features[i], dropout_rate)
        if use_attention:
            self.attention_gates = nn.ModuleDict()
            for i in range(len(features) - 1):
                self.attention_gates[f"att_0_{i}"] = AttentionGate(F_g=features[i], F_l=features[i], F_int=features[i] // 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        for i, (encoder, pool) in enumerate(zip(self.encoders[:-1], self.pools)):
            x = encoder(x)
            encoder_outputs.append(x)
            x = pool(x)
        x = self.encoders[-1](x)
        for decoder_level in range(len(self.features) - 2, -1, -1):
            x_up = self.ups[decoder_level](x)
            skip_connection = encoder_outputs[decoder_level]
            if self.use_attention:
                att_key = f"att_0_{decoder_level}"
                if att_key in self.attention_gates:
                    skip_connection = self.attention_gates[att_key](x_up, skip_connection)
            x = torch.cat([x_up, skip_connection], dim=1)
            conv_key = f"conv_0_{decoder_level}"
            if conv_key in self.decoder_convs:
                x = self.decoder_convs[conv_key](x)
        return self.final_conv(x)
