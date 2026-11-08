"""
STAGE 2: PURE CONVOLUTIONAL NEURAL NETWORK (CNN) ARCHITECTURE
-------------------------------------------------------------
Sequential implementation of the 7-stage CNN architecture:
1. Input Image: Single-channel Grayscale (1, 128, 128)
2. Convolution Layer + Batch Normalization: Feature extraction and internal activation stabilization
3. ReLU Layer: Non-linear activation function f(x) = max(0, x)
4. Pooling Layer: Downsampling via Max Pooling (2x2) for spatial invariance
5. Flatten Layer: Converts 3D feature tensor into 1D feature vector
6. Fully Connected Layer: Dense neural network with Dropout (0.4) for regularization
7. Output Layer: Logits mapped to class probabilities via Softmax (defect vs normal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PureCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(PureCNN, self).__init__()
        
        # -------------------------------------------------------------
        # BLOCK 1: Low-level feature extraction (Edges, Contours, Gradients)
        # Input: (B, 1, 128, 128) -> Output: (B, 16, 64, 64)
        # -------------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)  # Stabilizes layer activations and prevents oscillations
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # -------------------------------------------------------------
        # BLOCK 2: Mid-level feature extraction (Surface textures, Curves)
        # Input: (B, 16, 64, 64) -> Output: (B, 32, 32, 32)
        # -------------------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # -------------------------------------------------------------
        # BLOCK 3: High-level feature extraction (Complex defect patterns)
        # Input: (B, 32, 32, 32) -> Output: (B, 64, 16, 16)
        # -------------------------------------------------------------
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # -------------------------------------------------------------
        # Stage 5: Flatten Layer
        # Converts 3D tensor (64 x 16 x 16 = 16,384) into 1D vector
        # -------------------------------------------------------------
        self.flatten = nn.Flatten()
        
        # -------------------------------------------------------------
        # Stage 6: Fully Connected (Dense) Layer
        # Connects all extracted visual features to hidden decision neurons
        # -------------------------------------------------------------
        self.fc1 = nn.Linear(in_features=64 * 16 * 16, out_features=128)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)  # Regularization to prevent overfitting
        
        # -------------------------------------------------------------
        # Stage 7: Output Classification Layer
        # Outputs raw scores (logits) for each class: [defect_ring, normal_ring]
        # -------------------------------------------------------------
        self.fc_out = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        """
        Executes the sequential forward pass of the CNN.
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Flattening
        x = self.flatten(x)
        
        # Dense classification layers
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        
        # Raw logits output
        logits = self.fc_out(x)
        return logits

    def predict_proba(self, x):
        """
        Applies Softmax to convert raw logits into probability distribution (0.0 to 1.0).
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities


def inspect_architecture():
    """
    Verification function: simulates a forward pass with dummy input
    and prints the transformation of tensor dimensions across all stages.
    """
    print("=== [STAGE 2] PURE CNN ARCHITECTURE VERIFICATION ===")
    
    model = PureCNN(in_channels=1, num_classes=2)
    model.eval()
    
    dummy_input = torch.randn(1, 1, 128, 128)
    
    print("\n--- Layer-by-Layer Dimension Transformation ---")
    print(f"0. Input Image                 : {dummy_input.shape}")
    
    x = model.conv1(dummy_input)
    x = model.bn1(x)
    print(f"1. After Conv1+BatchNorm (16)  : {x.shape}")
    x = model.relu1(x)
    x = model.pool1(x)
    print(f"2. After MaxPool1 (2x2)        : {x.shape}")
    
    x = model.conv2(x)
    x = model.bn2(x)
    print(f"3. After Conv2+BatchNorm (32)  : {x.shape}")
    x = model.relu2(x)
    x = model.pool2(x)
    print(f"4. After MaxPool2 (2x2)        : {x.shape}")
    
    x = model.conv3(x)
    x = model.bn3(x)
    print(f"5. After Conv3+BatchNorm (64)  : {x.shape}")
    x = model.relu3(x)
    x = model.pool3(x)
    print(f"6. After MaxPool3 (2x2)        : {x.shape}")
    
    x = model.flatten(x)
    print(f"7. After Flatten (1D Vector)   : {x.shape} (64 x 16 x 16 = {x.shape[1]} features)")
    
    x = model.fc1(x)
    x = model.relu_fc(x)
    print(f"8. After Fully Connected (128) : {x.shape}")
    
    logits = model.fc_out(x)
    print(f"9. Output Logits (2 Classes)   : {logits.shape}")
    
    probs = F.softmax(logits, dim=1)
    print(f"10. Output Softmax Probs       : {probs.squeeze().tolist()} (Sum = {probs.sum().item():.2f})")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n-> Total Trainable Parameters  : {total_params:,} weights")
    print("=== [STAGE 2] SUCCESS: Pure CNN model architecture is verified and ready for training ===")

if __name__ == "__main__":
    inspect_architecture()
