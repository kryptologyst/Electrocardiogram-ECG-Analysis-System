"""Advanced ECG analysis models."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGCNN(nn.Module):
    """1D CNN for ECG signal classification (baseline model)."""
    
    def __init__(
        self,
        input_length: int = 250,
        num_classes: int = 4,
        dropout_rate: float = 0.5,
    ):
        """Initialize ECG CNN.
        
        Args:
            input_length: Length of input ECG signal.
            num_classes: Number of output classes.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            dummy_output = self._forward_features(dummy_input)
            flattened_size = dummy_output.numel()
        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers."""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ECGResNet(nn.Module):
    """1D ResNet for ECG signal classification."""
    
    def __init__(
        self,
        input_length: int = 250,
        num_classes: int = 4,
        dropout_rate: float = 0.5,
    ):
        """Initialize ECG ResNet.
        
        Args:
            input_length: Length of input ECG signal.
            num_classes: Number of output classes.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def _make_layer(
        self, 
        in_channels: int, 
        out_channels: int, 
        blocks: int, 
        stride: int = 1
    ) -> nn.Module:
        """Make a ResNet layer."""
        layers = []
        
        # First block with potential downsampling
        layers.append(
            ResidualBlock(in_channels, out_channels, stride)
        )
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(
                ResidualBlock(out_channels, out_channels)
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for 1D ResNet."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1
    ):
        """Initialize residual block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for convolution.
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, 
                    stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class ECGTransformer(nn.Module):
    """Transformer-based model for ECG signal classification."""
    
    def __init__(
        self,
        input_length: int = 250,
        num_classes: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout_rate: float = 0.1,
    ):
        """Initialize ECG Transformer.
        
        Args:
            input_length: Length of input ECG signal.
            num_classes: Number of output classes.
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            dropout_rate: Dropout rate.
        """
        super().__init__()
        
        self.d_model = d_model
        self.input_length = input_length
        
        # Input projection
        self.input_projection = nn.Conv1d(1, d_model, kernel_size=1)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout_rate)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input projection: (B, 1, L) -> (B, d_model, L)
        x = self.input_projection(x)
        
        # Transpose for transformer: (B, d_model, L) -> (B, L, d_model)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, d_model)
        
        # Classification
        x = self.classifier(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ECGEnsemble(nn.Module):
    """Ensemble of multiple ECG models."""
    
    def __init__(
        self,
        models: list,
        weights: Optional[list] = None,
    ):
        """Initialize ensemble model.
        
        Args:
            models: List of trained models.
            weights: Optional weights for each model.
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        outputs = []
        
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average
        ensemble_output = torch.zeros_like(outputs[0])
        for output, weight in zip(outputs, self.weights):
            ensemble_output += weight * output
        
        return ensemble_output
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
