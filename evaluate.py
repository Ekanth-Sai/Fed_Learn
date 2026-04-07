import os 
import argparse 
import torch 
import torch.nn as nn
from tqdm import tqdm 

from src.model import build_model 
from src.dataset import get_client_dataloader 
from src.utils import (
    plot_confusion_matrix,
    print_classification_report
)

def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    y_true = []
    y_pred = []
    total_loss = 0.0 
    total_samples = 0 
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc = "Evaluating", unit = "batch"):
            images = images.to(device)
            labels = labels.to(device) 
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1) 
            
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0) 
            
    avg_loss = total_loss / total_samples 
    accuracy = sum(t == p  for t, p in zip(y_true, y_pred)) / total_samples * 100  
    
    return y_true, y_pred, avg_loss, accuracy 

def main(args):
    device = torch.device("cpu")
    os.makedirs(args.output_dir, exist_ok = True)
    
    print(f"\n[Evaluate] Loading model from: {args.model_path}")
    model = build_model(num_classes = 7, pretrained = False)
    state_dict = torch.load(args.model_path, map_location = device)
    model.load_state_dict(state_dict)
    model.to(device)
    print("[Evaluate] Model loaded successfully") 
    
    print(f"[Evaluate] Loading test set from {args.test_csv}")
    test_loader = get_client_dataloader(
        csv_path = args.test_csv,
        batch_size = 32,
        train = False
    )
    print(f"[Evaluate] Test samples: {len(test_loader.dataset)}")
    
    print("\n[Evaluate] Running evaluation...")
    y_true, y_pred, avg_loss, accuracy = evaluate_model(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results: ")
    print(f"{'='*40}")
    print(f"Test loss: {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")
    
    print_classification_report(y_true, y_pred) 
    
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, save_path = cm_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Evaluate the trained global FL model on the test set"
    )
    
    parser.add_argument(
        "--model_path",
        type = str,
        default = "outputs/best_global_model.pth",
        help = "Path to saved model .pth file"
    )
    
    parser.add_argument(
        "--test_csv",
        type = str,
        default = "client_data/test.csv",
        help = "Path to test set CSV"
    )
    
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "outputs",
        help = "Where to save confusion matrix image"
    )
    
    args = parser.parse_args()
    main(args)