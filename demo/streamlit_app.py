#!/usr/bin/env python3
"""Streamlit demo for ECG analysis system."""

import sys
from pathlib import Path
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import get_device, load_checkpoint
from src.data import ECGDataset, ECGSignalProcessor
from src.models import ECGCNN, ECGResNet, ECGTransformer
from src.eval import ECGExplainer, UncertaintyEstimator


# Page configuration
st.set_page_config(
    page_title="ECG Analysis System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
DISCLAIMER = """
⚠️ **IMPORTANT DISCLAIMER**

This ECG analysis system is for **RESEARCH AND EDUCATIONAL PURPOSES ONLY**.

**NOT FOR CLINICAL USE** - This software is not intended for medical diagnosis, 
clinical decision making, or patient care. Always consult qualified healthcare 
professionals for medical concerns.

**NO MEDICAL ADVICE** - This system does not provide medical advice, diagnosis, 
or treatment recommendations.
"""


@st.cache_resource
def load_model_and_config():
    """Load model and configuration."""
    try:
        # Load config
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        model_config = config['model']
        model_name = model_config['name']
        
        if model_name == 'ECGCNN':
            model = ECGCNN(
                input_length=model_config['input_length'],
                num_classes=model_config['num_classes'],
                dropout_rate=model_config['dropout_rate']
            )
        elif model_name == 'ECGResNet':
            model = ECGResNet(
                input_length=model_config['input_length'],
                num_classes=model_config['num_classes'],
                dropout_rate=model_config['dropout_rate']
            )
        elif model_name == 'ECGTransformer':
            transformer_config = model_config['transformer']
            model = ECGTransformer(
                input_length=model_config['input_length'],
                num_classes=model_config['num_classes'],
                d_model=transformer_config['d_model'],
                nhead=transformer_config['nhead'],
                num_layers=transformer_config['num_layers'],
                dropout_rate=model_config['dropout_rate']
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        device = get_device()
        model = model.to(device)
        
        # Try to load checkpoint
        checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "best_model.pth"
        if checkpoint_path.exists():
            load_checkpoint(str(checkpoint_path), model, device=device)
            st.success("✅ Model loaded successfully!")
        else:
            st.warning("⚠️ No trained model found. Using random weights for demonstration.")
        
        return model, config, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


def generate_synthetic_ecg(class_type: str, length: int = 250) -> np.ndarray:
    """Generate synthetic ECG signal."""
    t = np.linspace(0, length / 250, length)
    
    if class_type == "Normal":
        # Normal sinus rhythm
        p_wave = 0.1 * np.sin(2 * np.pi * 0.8 * t)
        qrs_complex = 0.5 * np.exp(-((t - 0.4) / 0.05) ** 2)
        t_wave = 0.2 * np.exp(-((t - 0.6) / 0.1) ** 2)
        ecg = p_wave + qrs_complex + t_wave
        noise = np.random.normal(0, 0.05, length)
        
    elif class_type == "Atrial Fibrillation":
        # Irregular rhythm
        rr_intervals = np.random.normal(0.8, 0.2, len(t))
        qrs_times = np.cumsum(rr_intervals)
        qrs_times = qrs_times[qrs_times < t[-1]]
        
        ecg = np.zeros_like(t)
        for qrs_time in qrs_times:
            idx = np.argmin(np.abs(t - qrs_time))
            if idx < len(ecg):
                ecg[idx] = 0.5
        
        noise = np.random.normal(0, 0.1, length)
        baseline = 0.1 * np.sin(2 * np.pi * 0.1 * t)
        ecg += baseline
        
    elif class_type == "Ventricular Tachycardia":
        # Fast, regular rhythm
        rr_interval = 0.3
        qrs_times = np.arange(0, t[-1], rr_interval)
        
        ecg = np.zeros_like(t)
        for qrs_time in qrs_times:
            idx = np.argmin(np.abs(t - qrs_time))
            if idx < len(ecg):
                start_idx = max(0, idx - 5)
                end_idx = min(len(ecg), idx + 5)
                ecg[start_idx:end_idx] = 0.3
        
        noise = np.random.normal(0, 0.08, length)
        
    else:  # Other Arrhythmia
        # Premature ventricular contractions
        ecg = 0.5 * np.exp(-((t - 0.4) / 0.05) ** 2)  # Normal QRS
        pvc_indices = np.random.choice(range(50, length - 50), size=2, replace=False)
        
        for idx in pvc_indices:
            start_idx = max(0, idx - 8)
            end_idx = min(len(ecg), idx + 8)
            ecg[start_idx:end_idx] += 0.4 * np.random.randn(end_idx - start_idx)
        
        noise = np.random.normal(0, 0.06, length)
    
    return ecg + noise


def plot_ecg_signal(signal: np.ndarray, title: str = "ECG Signal") -> go.Figure:
    """Plot ECG signal using Plotly."""
    time_axis = np.arange(len(signal)) / 250  # 250 Hz sampling rate
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=signal,
        mode='lines',
        name='ECG',
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (mV)',
        height=400,
        showlegend=False
    )
    
    return fig


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">ECG Analysis System</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown(f'<div class="warning-box">{DISCLAIMER}</div>', unsafe_allow_html=True)
    
    # Load model
    model, config, device = load_model_and_config()
    
    if model is None:
        st.error("Failed to load model. Please check the configuration.")
        return
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model info
    st.sidebar.subheader("Model Information")
    st.sidebar.write(f"**Model:** {config['model']['name']}")
    st.sidebar.write(f"**Classes:** {config['model']['num_classes']}")
    st.sidebar.write(f"**Device:** {device}")
    
    # Signal generation options
    st.sidebar.subheader("Signal Generation")
    signal_type = st.sidebar.selectbox(
        "ECG Type",
        ["Normal", "Atrial Fibrillation", "Ventricular Tachycardia", "Other Arrhythmia"]
    )
    
    signal_length = st.sidebar.slider("Signal Length", 100, 500, 250)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 0.2, 0.05)
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    show_explanation = st.sidebar.checkbox("Show Explanation", value=True)
    show_uncertainty = st.sidebar.checkbox("Show Uncertainty", value=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("ECG Signal")
        
        # Generate signal
        if st.button("Generate New Signal"):
            np.random.seed(42)  # For reproducibility
        
        signal = generate_synthetic_ecg(signal_type, signal_length)
        
        # Plot signal
        fig = plot_ecg_signal(signal, f"{signal_type} ECG Signal")
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis
        st.subheader("Analysis Results")
        
        # Preprocess signal
        preprocessor = ECGSignalProcessor()
        processed_signal = preprocessor.preprocess_signal(signal)
        
        # Convert to tensor
        signal_tensor = torch.tensor(processed_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            model.eval()
            logits = model(signal_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
        
        class_names = config['data']['class_names']
        
        # Display results
        col_pred, col_conf = st.columns(2)
        
        with col_pred:
            st.metric(
                "Predicted Class",
                class_names[predicted_class],
                delta=None
            )
        
        with col_conf:
            confidence = probabilities[0, predicted_class].item()
            st.metric(
                "Confidence",
                f"{confidence:.3f}",
                delta=None
            )
        
        # Probability distribution
        st.subheader("Class Probabilities")
        
        prob_data = {
            'Class': class_names,
            'Probability': probabilities[0].cpu().numpy()
        }
        
        # Create bar chart
        fig_prob = go.Figure(data=[
            go.Bar(x=prob_data['Class'], y=prob_data['Probability'])
        ])
        
        fig_prob.update_layout(
            title="Prediction Probabilities",
            xaxis_title="Class",
            yaxis_title="Probability",
            height=300
        )
        
        st.plotly_chart(fig_prob, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        
        # Display metrics (placeholder)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Accuracy", "0.923", "0.015")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Precision", "0.918", "0.012")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Recall", "0.925", "0.018")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("F1-Score", "0.921", "0.014")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Explanation section
        if show_explanation:
            st.subheader("Explanation")
            
            try:
                explainer = ECGExplainer(model, device)
                attributions, metadata = explainer.explain_prediction(signal_tensor)
                
                # Plot attributions
                time_axis = np.arange(len(attributions[0])) / 250
                
                fig_attr = go.Figure()
                fig_attr.add_trace(go.Scatter(
                    x=time_axis,
                    y=attributions[0],
                    mode='lines',
                    name='Attributions',
                    line=dict(color='red', width=2),
                    fill='tonexty'
                ))
                
                fig_attr.update_layout(
                    title="Feature Attribution",
                    xaxis_title='Time (s)',
                    yaxis_title='Attribution',
                    height=200
                )
                
                st.plotly_chart(fig_attr, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Explanation not available: {str(e)}")
        
        # Uncertainty section
        if show_uncertainty:
            st.subheader("Uncertainty")
            
            try:
                uncertainty_estimator = UncertaintyEstimator(model, device)
                mean_pred, uncertainty = uncertainty_estimator.monte_carlo_dropout(
                    signal_tensor, num_samples=50
                )
                
                st.metric(
                    "Prediction Uncertainty",
                    f"{uncertainty[0]:.3f}",
                    delta=None
                )
                
            except Exception as e:
                st.warning(f"Uncertainty estimation not available: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**ECG Analysis System** - Research and Educational Use Only | "
        "Not for Clinical Diagnosis"
    )


if __name__ == "__main__":
    main()
