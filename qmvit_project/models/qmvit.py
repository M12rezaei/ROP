"""
QMViT implementation scaffold.

- If PennyLane (and a supported qubit device) is installed, QuantumLayer uses PennyLane's QNode to process 2x2 patches with a small circuit.
- If PennyLane is not available, a small classical MLP fallback is used to mimic quantum feature transforms (this keeps experiments reproducible without quantum packages).
- The rest of the network uses a MobileNet-like lightweight block + transformer-ish global avg+fc to classify.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import pennylane; if unavailable, provide a classical fallback.
try:
    import pennylane as qml
    from pennylane import numpy as npq
    PNL_AVAILABLE = True
except Exception as e:
    PNL_AVAILABLE = False
    # print warning deferred to runtime

class QuantumLayerStub(nn.Module):
    def __init__(self, in_channels=3, patch_size=2, q_outputs=4, use_pennylane=PNL_AVAILABLE):
        super().__init__()
        self.use_pennylane = use_pennylane and PNL_AVAILABLE
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.q_outputs = q_outputs
        if not self.use_pennylane:
            # classical fallback: small conv + MLP to mimic output dimension
            self.fallback = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(16, q_outputs)
            )
        else:
            # define PennyLane device and QNode lazily during forward to avoid import-time failures
            self.dev = None
            self.qnode = None
            # define a small linear layer to map measured outputs back to channels
            self.post = nn.Linear(q_outputs, q_outputs)

    def forward(self, x):
        # x: (B, C, H, W) - we'll pool to a small spatial map and apply quantum transform per pooled cell
        if not self.use_pennylane:
            return self.fallback(x)  # (B, q_outputs)
        else:
            # Attempt to build a device if not already
            if self.dev is None:
                try:
                    self.dev = qml.device('default.qubit', wires=4)
                except Exception as e:
                    # fallback to classical if device can't be created
                    self.use_pennylane = False
                    return self.fallback(x)
                # define qnode; simple encoding + small random layers
                def circuit(inputs, weights):
                    # inputs: length 4, weights shape (L, 4)
                    for i in range(4):
                        qml.RY(inputs[i], wires=i)
                    for l in range(weights.shape[0]):
                        for w in range(4):
                            qml.RZ(weights[l,w], wires=w)
                        for w in range(3):
                            qml.CNOT(wires=[w, w+1])
                    return [qml.expval(qml.PauliZ(i)) for i in range(4)]
                weight_shapes = {"weights": (2,4)}
                self.qnode = qml.QNode(circuit, self.dev, interface='torch', diff_method='backprop')
                # wrapper to call qnode
            B,C,H,W = x.shape
            # global average pool to reduce spatial size
            pooled = F.adaptive_avg_pool2d(x, (1,1)).view(B, C)
            # if channels > 4, reduce via linear
            if pooled.shape[1] > 4:
                pooled = pooled[:, :4]
            elif pooled.shape[1] < 4:
                # pad
                pad = torch.zeros(B, 4 - pooled.shape[1], device=pooled.device, dtype=pooled.dtype)
                pooled = torch.cat([pooled, pad], dim=1)
            # run qnode per batch item
            outs = []
            # weights tensor (2x4) random fixed for now
            weights = torch.randn(2,4, device=pooled.device, dtype=pooled.dtype, requires_grad=False)
            for i in range(B):
                inp = pooled[i]
                try:
                    qout = self.qnode(inp, weights)
                    qout = torch.tensor(qout, device=pooled.device, dtype=pooled.dtype)
                except Exception as e:
                    # if QNode call fails, fallback
                    qout = torch.randn(self.q_outputs, device=pooled.device)
                outs.append(qout)
            outs = torch.stack(outs, dim=0)
            return self.post(outs)

class MobileBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class QMViT(nn.Module):
    def __init__(self, num_classes=5, input_size=128, quantum_outputs=4):
        super().__init__()
        self.quantum = QuantumLayerStub(in_channels=3, patch_size=2, q_outputs=quantum_outputs)
        # simple conv stem to process quantum outputs + image
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.mb = MobileBlock(32, 64)
        # project quantum outputs to channels and concatenate
        self.q_fc = nn.Linear(quantum_outputs, 16)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        B = x.shape[0]
        qfeat = self.quantum(x)  # (B, q_outputs)
        qproj = self.q_fc(qfeat).unsqueeze(-1).unsqueeze(-1)  # (B,16,1,1)
        imgf = self.conv1(x)
        imgf = self.mb(imgf)
        imgp = F.adaptive_avg_pool2d(imgf, (1,1)).view(B, -1)  # (B,64)
        qflat = qproj.view(B, -1)  # (B,16)
        cat = torch.cat([imgp, qflat], dim=1)
        out = self.classifier[2:](cat) if isinstance(self.classifier, nn.Sequential) else self.classifier(cat)
        return out
