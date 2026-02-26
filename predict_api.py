# predict_api.py
from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort

app = Flask(__name__)
session = ort.InferenceSession("checkpoints/soh_model.onnx")

@app.route('/predict_soh', methods=['POST'])
def predict_soh():
    """
    Input: {"cycles": [{"temp": 25, "dod": 0.5, "is_fast_charge": 0, "cycle": 100}, ...]}
    Output: {"soh": 0.85}
    """
    try:
        data = request.json
        cycles = data['cycles']
        
        # 转为特征数组
        features = np.array([
            [c['temp'], c['dod'], c['is_fast_charge'], c['cycle']] 
            for c in cycles[-30:]  # 取最近30个cycle
        ], dtype=np.float32)
        
        # 补零到30长度（如果不足）
        if len(features) < 30:
            pad = np.zeros((30 - len(features), 4))
            features = np.vstack([pad, features])
        
        features = features.reshape(1, 30, 4)
        soh = session.run(None, {'input': features})[0][0]
        
        return jsonify({"soh": float(soh), "status": "success"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)