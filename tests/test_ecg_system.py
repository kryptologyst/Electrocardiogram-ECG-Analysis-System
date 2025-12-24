"""Tests for ECG analysis system."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import ECGCNN, ECGResNet, ECGTransformer
from src.data import ECGDataset, ECGSignalProcessor
from src.losses import FocalLoss, LabelSmoothingCrossEntropy, CombinedLoss
from src.metrics import ClinicalMetrics
from src.utils import set_seed, get_device, count_parameters


class TestModels:
    """Test model architectures."""
    
    def test_ecg_cnn(self):
        """Test ECGCNN model."""
        model = ECGCNN(input_length=250, num_classes=4)
        
        # Test forward pass
        x = torch.randn(2, 1, 250)
        output = model(x)
        
        assert output.shape == (2, 4)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_ecg_resnet(self):
        """Test ECGResNet model."""
        model = ECGResNet(input_length=250, num_classes=4)
        
        # Test forward pass
        x = torch.randn(2, 1, 250)
        output = model(x)
        
        assert output.shape == (2, 4)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_ecg_transformer(self):
        """Test ECGTransformer model."""
        model = ECGTransformer(
            input_length=250, 
            num_classes=4,
            d_model=64,
            nhead=4,
            num_layers=2
        )
        
        # Test forward pass
        x = torch.randn(2, 1, 250)
        output = model(x)
        
        assert output.shape == (2, 4)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_parameters(self):
        """Test model parameter counts."""
        cnn = ECGCNN(input_length=250, num_classes=4)
        resnet = ECGResNet(input_length=250, num_classes=4)
        transformer = ECGTransformer(input_length=250, num_classes=4)
        
        cnn_params = count_parameters(cnn)
        resnet_params = count_parameters(resnet)
        transformer_params = count_parameters(transformer)
        
        assert cnn_params > 0
        assert resnet_params > 0
        assert transformer_params > 0
        
        # ResNet should have more parameters than CNN
        assert resnet_params > cnn_params


class TestData:
    """Test data processing."""
    
    def test_ecg_dataset(self):
        """Test ECG dataset creation."""
        dataset = ECGDataset(num_samples=100, seq_length=250, num_classes=4)
        
        assert len(dataset) == 100
        
        # Test data loading
        signal, label = dataset[0]
        assert signal.shape == (1, 250)
        assert isinstance(label, int)
        assert 0 <= label < 4
    
    def test_signal_processor(self):
        """Test ECG signal processor."""
        processor = ECGSignalProcessor(sampling_rate=250)
        
        # Generate test signal
        signal = np.random.randn(250)
        
        # Test preprocessing
        processed = processor.preprocess_signal(signal)
        
        assert processed.shape == signal.shape
        assert not np.isnan(processed).any()
        assert not np.isinf(processed).any()
    
    def test_r_peak_detection(self):
        """Test R-peak detection."""
        processor = ECGSignalProcessor(sampling_rate=250)
        
        # Generate signal with clear peaks
        t = np.linspace(0, 1, 250)
        signal = np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(250)
        
        peaks = processor.detect_r_peaks(signal)
        
        assert len(peaks) > 0
        assert all(0 <= peak < len(signal) for peak in peaks)


class TestLosses:
    """Test loss functions."""
    
    def test_focal_loss(self):
        """Test focal loss."""
        loss_fn = FocalLoss(gamma=2.0)
        
        # Test forward pass
        inputs = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(inputs, targets)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_label_smoothing_loss(self):
        """Test label smoothing loss."""
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # Test forward pass
        inputs = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(inputs, targets)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_combined_loss(self):
        """Test combined loss."""
        loss_fn = CombinedLoss()
        
        # Test forward pass
        inputs = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(inputs, targets)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_clinical_metrics(self):
        """Test clinical metrics computation."""
        metrics = ClinicalMetrics(num_classes=3)
        
        # Generate test data
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])
        y_prob = np.random.rand(5, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Compute metrics
        results = metrics.compute_metrics(y_true, y_pred, y_prob)
        
        # Check that all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in expected_metrics:
            assert metric in results
            assert 0 <= results[metric] <= 1


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate random numbers
        torch_rand = torch.rand(5)
        np_rand = np.random.rand(5)
        
        # Set seed again and generate same numbers
        set_seed(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Should be the same
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = ECGCNN(input_length=250, num_classes=4)
        param_count = count_parameters(model)
        
        assert param_count > 0
        assert isinstance(param_count, int)


class TestIntegration:
    """Integration tests."""
    
    def test_training_loop(self):
        """Test basic training loop."""
        # Create model and data
        model = ECGCNN(input_length=250, num_classes=4)
        dataset = ECGDataset(num_samples=50, seq_length=250, num_classes=4)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Create optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for batch in dataloader:
            signals, labels = batch
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Check that loss is finite
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            break  # Only test one batch
    
    def test_evaluation_loop(self):
        """Test basic evaluation loop."""
        # Create model and data
        model = ECGCNN(input_length=250, num_classes=4)
        dataset = ECGDataset(num_samples=20, seq_length=250, num_classes=4)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        
        # Evaluation loop
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                signals, labels = batch
                outputs = model(signals)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Check outputs
                assert outputs.shape[0] == signals.shape[0]
                assert outputs.shape[1] == 4
                assert probabilities.shape == outputs.shape
                assert predictions.shape[0] == signals.shape[0]
                break  # Only test one batch


if __name__ == "__main__":
    pytest.main([__file__])
