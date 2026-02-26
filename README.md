# EV Battery State-of-Health Prediction with PyTorch Transformers

> **Designed for real-world EV deployment in China**  
> A lightweight SOH estimation system optimized for Chinese driving patterns and climate conditions.

![SOH Prediction](docs/soh_prediction.png)

## 🌏 Why China-Specific?
- **Fast-charging dominance**: 30% of cycles use DC fast charging → accelerated degradation
- **High-temperature stress**: 20% of cycles simulated at >35°C (Guangzhou/Shenzhen summer)
- **Shallow cycling**: 60% of cycles between 20%-80% SoC (typical ride-hailing pattern)

## 📊 Performance
| Metric | Value |
|--------|-------|
| MAE | 0.012 (1.2%) |
| Inference Latency | 15ms (CPU) |
| Model Size | 4.7 MB (ONNX) |

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data & train model
python train.py

# Run inference API
python predict_api.py

# Test prediction
curl -X POST http://localhost:5000/predict_soh -H "Content-Type: application/json" -d '{
  "cycles": [
    {"temp": 25, "dod": 0.5, "is_fast_charge": 0, "cycle": 98},
    {"temp": 38, "dod": 0.7, "is_fast_charge": 1, "cycle": 99},
    {"temp": 30, "dod": 0.6, "is_fast_charge": 0, "cycle": 100}
  ]
}'