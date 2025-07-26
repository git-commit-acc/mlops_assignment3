import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

sk_model = joblib.load('sk_model.joblib')
weights = sk_model.coef_
bias = sk_model.intercept_
params = {'coef': weights, 'intercept': bias}
joblib.dump(params, 'unquant_params.joblib')

def quantize_param(arr):
    mn, mx = arr.min(), arr.max()
    scale = (mx - mn) / 255.0 if (mx - mn) != 0 else 1.0
    q = np.round((arr - mn) / scale).astype(np.uint8)
    return q, mn, scale

q_weights, w_min, w_scale = quantize_param(weights)
q_bias, b_min, b_scale = quantize_param(np.array([bias]))
quantized = {
    'q_weights': q_weights, 'w_min': w_min, 'w_scale': w_scale,
    'q_bias': q_bias, 'b_min': b_min, 'b_scale': b_scale
}
joblib.dump(quantized, 'quant_params.joblib')

def dequantize_param(q_arr, mn, scale):
    return q_arr.astype(np.float32) * scale + mn

deq_weights = dequantize_param(q_weights, w_min, w_scale)
deq_bias = dequantize_param(q_bias, b_min, b_scale)[0]

class QuantLinear(nn.Module):
    def __init__(self, in_f):
        super().__init__()
        self.fc = nn.Linear(in_f, 1)
    def forward(self, x):
        return self.fc(x)

pt_model = QuantLinear(deq_weights.shape[0])
with torch.no_grad():
    pt_model.fc.weight.copy_(torch.from_numpy(deq_weights.reshape(1, -1)))
    pt_model.fc.bias.copy_(torch.tensor([deq_bias]))
pt_model.eval()

data = fetch_california_housing()
X, y = data.data, data.target
X_tensor = torch.from_numpy(X).float()
y_pred = pt_model(X_tensor).detach().numpy().flatten()
r2 = r2_score(y, y_pred)

print(f'Quantized PyTorch model R^2: {r2}')
import os
print(f"Original sklearn model size: {os.path.getsize('sk_model.joblib')/1024:.1f} KB")
print(f"Quantized params size: {os.path.getsize('quant_params.joblib')/1024:.1f} KB")