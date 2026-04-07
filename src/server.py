import torch 
import torch.nn as nn
from torch.utils.data import DataLoader 
import copy 
from typing import List, Dict 

from src.model import build_model, get_model_parameters, set_model_parameters 
from src.dataset import get_client_dataloader 

class FederatedServer:
    def __init__(self, num_classes: int = 7, test_csv: str = None, device: str = "cpu"):
        self.num_classes = num_classes 
        self.device = torch.device(device)
        
        self.global_model = build_model(num_classes = num_classes, pretrained = False) 
        self.global_model.to(self.device) 
        
        self.criterion = nn.CrossEntropyLoss() 
        
        self.test_loader = None 
        
        if test_csv:
            self.test_loader = get_client_dataloader(csv_path = test_csv, batch_size = 32, train = False)
            print(f"[Server] Test set loaded | "
                    f"Samples: {len(self.test_loader.dataset)}")
            
        print(f"[Server] Global model initialized on {device}")
        
    def get_global_parameters(self) -> list:
        return get_model_parameters(self.global_model)
        
    def aggregate(self, client_results: List[Dict]) -> None:
        total_samples = sum(result["num_samples"] for result in client_results)
        global_param = self.get_global_parameters()
        aggregated = [torch.zeros_like(param) for param in global_param]
        for result in client_results:
            weight = result["num_samples"] / total_samples 
            client_params = result["parameters"]
            
            for i, client_param in enumerate(client_params):
                aggregated[i] += weight * client_param.to(self.device)
        
        set_model_parameters(self.global_model, aggregated)
        
        print(f"[Server] FedAvg complete | "
                f"Total samples used: {total_samples} | "
                f"Clients: {len(client_results)}")
        
    def evaluate(self) -> Dict:
        if self.test_loader is None:
            return {"test_loss": None, "test_acc": None}
        
        self.global_model.eval() 
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0 
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.global_model(images)
                loss = self.criterion(outputs, labels) 
                
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item() 
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0) 
                
        test_loss = total_loss / total_samples 
        test_acc = total_correct / total_samples * 100 
        
        return {"test_loss": test_loss, "test_acc": test_acc}
    
    def save_model(self, path: str) -> None:
        torch.save(self.global_model.state_dict(), path)
        print(f"[Server] Global model saved -> {path}")
        
    def load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location = self.device)
        self.global_model.load_state_dict(state_dict)
        print(f"[Server] Global model loaded <- {path}")