"""
PyTorch Model Definition for URFD RGB Baseline
Architecture: Frozen ImageNet-pretrained ResNet-18 + Temporal Mean-Std Pooling + MLP Classifier
"""

import torch
import torch.nn as nn
import torchvision.models as models

class URFDRGBBaseline(nn.Module):
    """
    URFD RGB Baseline Model.
    Consumes temporal windows of shape (B, T, C, H, W) = (B, 50, 3, 240, 320),
    extracts per-frame 512-dim features via frozen ResNet-18, computes temporal
    mean & std pooling (1024-dim), and passes through a 2-layer MLP classifier.
    """
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.dropout_p = dropout_p
        
        # Load ImageNet-pretrained ResNet-18 backbone
        try:
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
        except Exception as e:
            raise RuntimeError(
                f"CRITICAL ERROR: Failed to load ImageNet-pretrained ResNet-18 weights: {e}\n"
                "Pretrained weights are required for this baseline experiment. Randomly initialized weights "
                "are prohibited to prevent corrupting the baseline benchmark."
            )
            
        # Replace FC layer with Identity to extract 512-dim feature vectors
        resnet.fc = nn.Identity()
        self.feature_extractor = resnet
        
        # Freeze ALL backbone parameters strictly
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Trainable Classifier Head: 1024 -> 64 -> 2
        self.classifier = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        """
        Forward pass.
        Input x shape: (B, T, C, H, W) = (B, 50, 3, 240, 320)
        Output shape: (B, 2) unnormalized logits
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input tensor (B, T, C, H, W), got {x.dim()}D shape {x.shape}")
            
        B, T, C, H, W = x.shape
        x_reshaped = x.view(B * T, C, H, W)
        
        # Extract features under torch.no_grad context for frozen backbone
        with torch.no_grad():
            feats = self.feature_extractor(x_reshaped) # Shape: (B*T, 512)
            
        feats = feats.view(B, T, 512) # Shape: (B, T, 512)
        
        # Temporal Mean and Standard Deviation Pooling across T=50 frames
        mean_feat = torch.mean(feats, dim=1) # (B, 512)
        std_feat = torch.std(feats, dim=1)   # (B, 512)
        
        # Concatenate into 1024-dim representation
        pooled = torch.cat([mean_feat, std_feat], dim=1) # (B, 1024)
        
        # Classification Head
        logits = self.classifier(pooled) # (B, 2)
        return logits

    def get_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": frozen}


class URFDRGBFeatureBaseline(nn.Module):
    """
    Feature-based URFD RGB Baseline Model.
    Consumes precomputed per-frame 512-dim features of shape (B, T, 512) = (B, 50, 512),
    computes temporal mean & std pooling (1024-dim), and passes through a 2-layer MLP classifier.
    Contains ONLY trainable classifier parameters (65,730 parameters).
    """
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.dropout_p = dropout_p
        
        # Trainable Classifier Head: 1024 -> 64 -> 2
        self.classifier = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        """
        Forward pass.
        Input x shape: (B, T, 512) = (B, 50, 512)
        Output shape: (B, 2) unnormalized logits
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input feature tensor (B, T, 512), got {x.dim()}D shape {x.shape}")
            
        B, T, D = x.shape
        if D != 512:
            raise ValueError(f"Expected feature dimension 512, got {D}")

        # Temporal Mean and Standard Deviation Pooling across T=50 frames
        mean_feat = torch.mean(x, dim=1) # (B, 512)
        std_feat = torch.std(x, dim=1)   # (B, 512)
        
        # Concatenate into 1024-dim representation
        pooled = torch.cat([mean_feat, std_feat], dim=1) # (B, 1024)
        
        # Classification Head
        logits = self.classifier(pooled) # (B, 2)
        return logits

    def get_parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": frozen}


if __name__ == "__main__":
    # Model Setup & Dimension Test
    print("=== TESTING URFDRGBBaseline MODEL ARCHITECTURE ===")
    model = URFDRGBBaseline(dropout_p=0.5)
    counts = model.get_parameter_counts()
    
    print(f"Total Parameters: {counts['total']:,}")
    print(f"Frozen Backbone Parameters: {counts['frozen']:,}")
    print(f"Trainable Classifier Parameters: {counts['trainable']:,}")
    
    # Test Dummy Forward Pass
    dummy_input = torch.randn(2, 50, 3, 240, 320)
    output = model(dummy_input)
    print(f"\nDummy Forward Pass Successful!")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Logits Shape: {output.shape} (Expected [2, 2])")
