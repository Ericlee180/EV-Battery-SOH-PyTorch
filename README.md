# EV Battery State-of-Health Prediction with PyTorch Transformers

> **Designed for real-world EV deployment in China**  
> A lightweight SOH estimation system optimized for Chinese driving patterns and climate conditions.

## 🌏 Why China-Specific?
- **Fast-charging dominance**: 70% of Chinese EV users rely on DC fast charging (vs. 30% in EU) → accelerated degradation
- **Extreme temperatures**: From -30°C (Harbin) to +45°C (Guangzhou) → thermal stress modeling critical
- **Ride-hailing fleets**: Shallow cycling (20%-80% SoC) dominates urban usage

## 📊 Performance
| Metric | Value | Test Condition |
|--------|-------|----------------|
| MAE | 1.8% | NASA Dataset (Group 5) |
| Inference Latency | 42ms | NVIDIA Jetson Nano |
| Model Size | 8.2 MB | ONNX format |

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python train.py --china_mode  # Enables China-specific feature engineering
python predict_api.py  # Run REST API