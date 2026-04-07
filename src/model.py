import torch 
import torch.nn as nn 
from torchvision import models 

def build_model(num_classes: int = 7, pretrained: bool = False) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None 
    model = models.mobilenet_v2(weights = weights)
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

def get_model_parameters(model: nn.Module) -> list:
    return [param.data.clone() for param in model.parameters()]

def set_model_parameters(model: nn.Module, parameters: list) -> None: 
    for param, new_param in zip(model.parameters(), parameters):
        param.data.copy_(new_param)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

if __name__ == "__main__":
    model = build_model(num_classes = 7, pretrained = False)
    print(f"Model architecture:\n{model}\n")
    print(f"Total trainable parameters: {count_parameters(model):,}")
    
    dummy_input = torch.randn(4, 3, 224, 224) 
    output = model(dummy_input)
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape}")
    print("Model builds and forward pass working")