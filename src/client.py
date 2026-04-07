import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
import copy 
import pandas as pd

from src.dataset import get_client_dataloader 
from src.model import build_model, set_model_parameters, get_model_parameters 

class FederatedClient:
    def __init__(self, client_id: int, csv_path: str, num_classes: int = 7, local_epochs: int = 2, batch_size: int = 16, lr: float = 1e-3, device: str = "cpu"):
        self.client_id = client_id
        self.csv_path = csv_path
        self.num_classes = num_classes 
        self.local_epochs = local_epochs 
        self.batch_size = batch_size 
        self.lr = lr 
        self.device = torch.device(device)
        self.model = build_model(num_classes = num_classes, pretrained = False)
        self.model.to(self.device)
        
        # self.criterion = nn.CrossEntropyLoss() 
        
        class_counts = pd.read_csv(csv_path)['dx'].value_counts()
        classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        counts = torch.tensor([class_counts.get(c, 1) for c in classes], dtype=torch.float)
        weights = 1.0 / counts
        weights = weights / weights.sum()
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        
        self.dataloader = get_client_dataloader(csv_path = csv_path, batch_size = batch_size, train = True)
        self.num_samples = len(self.dataloader.dataset) 
        
        print(f"[Client {client_id}] Initialized | "
                f"Samples: {self.num_samples} | "
                f"Batches per epoch: {len(self.dataloader)}")
    
    def receive_global_model(self, global_parameters: list) -> None:
        set_model_parameters(self.model, global_parameters)
        
    def local_train(self, current_lr: float) -> dict:
        optimizer = optim.SGD(self.model.parameters(), lr = current_lr, momentum = 0.9, weight_decay = 1e-4)
        
        self.model.train() 
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0 
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0 
            epoch_correct = 0
            epoch_samples = 0 
            
            for batch_idx, (images, labels) in enumerate(self.dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device) 
                
                optimizer.zero_grad() 
                outputs = self.model(images)
                loss = self.criterion(outputs, labels) 
                
                loss.backward() 
                optimizer.step()
                
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(labels).sum().item() 
                batch_size = labels.size(0)
                
                epoch_loss += loss.item() * batch_size
                epoch_correct += batch_correct 
                epoch_samples += batch_size
                
            avg_epoch_loss = epoch_loss / epoch_samples 
            avg_epoch_acc = epoch_correct / epoch_samples * 100 
            
            total_loss += epoch_loss 
            total_correct += epoch_correct 
            total_samples += epoch_samples 
            
            print(f"[Client {self.client_id}] Epoch {epoch + 1} / {self.local_epochs} "
                    f"| Loss: {avg_epoch_loss:.4f} | Acc: {avg_epoch_acc:.2f}%")
            
        avg_train_loss = total_loss / total_samples
        avg_train_acc = total_correct / total_samples * 100 
        
        return {
            "parameters": get_model_parameters(self.model),
            "num_samples": self.num_samples,
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc
        }

    def train_one_round(self, global_parameters: list, current_lr: float) -> dict:
        self.receive_global_model(global_parameters)
        result = self.local_train(current_lr) 
        
        return result