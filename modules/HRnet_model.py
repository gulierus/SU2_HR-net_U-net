"""
HRNet (High-Resolution Network) implementation for cell segmentation.
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic residual block for HRNet."""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class HighResolutionModule(nn.Module):
    """High-Resolution Module with parallel branches and multi-scale fusion."""
    
    def __init__(self, num_branches, num_blocks, num_channels, dropout_rate=0.0):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        
        # Create parallel branches (each with different resolution)
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            self.branches.append(
                self._make_branch(num_blocks[i], num_channels[i], dropout_rate)
            )
        
        # Multi-scale fusion between branches
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if j > i:
                    # Upsampling (from lower resolution)
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        nn.BatchNorm2d(num_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                    ))
                elif j == i:
                    # Same resolution
                    fuse_layer.append(None)
                else:
                    # Downsampling (from higher resolution)
                    conv_list = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_list.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[i], 
                                         3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[i])
                            ))
                        else:
                            conv_list.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[j], 
                                         3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[j]),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv_list))
            self.fuse_layers.append(fuse_layer)
        
        self.relu = nn.ReLU(inplace=True)
    
    def _make_branch(self, num_blocks, num_channels, dropout_rate):
        """Create one branch with several blocks."""
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(num_channels, num_channels, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x is a list of tensors (one for each branch)
        
        # Process each branch independently
        x_branches = []
        for i in range(self.num_branches):
            x_branches.append(self.branches[i](x[i]))
        
        # Fuse information between branches
        x_fused = []
        for i in range(self.num_branches):
            y = x_branches[0] if i == 0 else self.fuse_layers[i][0](x_branches[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x_branches[j]
                else:
                    y = y + self.fuse_layers[i][j](x_branches[j])
            x_fused.append(self.relu(y))
        
        return x_fused


class HRNet(nn.Module):
    """
    HRNet for semantic segmentation.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        out_channels: Number of output channels (default: 1 for binary segmentation)
        base_channels: Base number of channels (default: 32)
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, dropout_rate=0.1):
        super(HRNet, self).__init__()
        
        # Stem (initial processing with downsampling to 1/4)
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1 (single branch, high resolution)
        self.layer1 = self._make_layer(64, base_channels, 4, dropout_rate)
        
        # Transition to 2 branches
        self.transition1 = nn.ModuleList([
            nn.Sequential(),  # High resolution (no change)
            nn.Sequential(    # Lower resolution (1/2)
                nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Stage 2 (2 branches)
        self.stage2 = HighResolutionModule(
            num_branches=2,
            num_blocks=[4, 4],
            num_channels=[base_channels, base_channels * 2],
            dropout_rate=dropout_rate
        )
        
        # Transition to 3 branches
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # High resolution (no change)
            nn.Sequential(),  # Medium resolution (no change)
            nn.Sequential(    # Low resolution (1/4)
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Stage 3 (3 branches)
        self.stage3 = HighResolutionModule(
            num_branches=3,
            num_blocks=[4, 4, 4],
            num_channels=[base_channels, base_channels * 2, base_channels * 4],
            dropout_rate=dropout_rate
        )
        
        # Final layer for segmentation (only from high-resolution branch)
        self.final_layer = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 1)
        )
        
        # Upsample to return to original resolution (4x upsampling)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, dropout_rate):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, dropout_rate=dropout_rate))
        for i in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Stem (downsample to 1/4)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Stage 1
        x = self.layer1(x)
        
        # Transition to 2 branches
        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        
        # Stage 2 (parallel processing + fusion)
        x_list = self.stage2(x_list)
        
        # Transition to 3 branches
        x_list_new = []
        for i in range(3):
            if i < 2:
                x_list_new.append(x_list[i])
            else:
                x_list_new.append(self.transition2[i](x_list[-1]))
        x_list = x_list_new
        
        # Stage 3 (parallel processing + fusion)
        x_list = self.stage3(x_list)
        
        # Take only high-resolution branch
        x = x_list[0]
        
        # Final prediction
        x = self.final_layer(x)
        x = self.upsample(x)
        
        return x


