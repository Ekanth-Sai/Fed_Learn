import os 
import json 
import random 
import numpy as np 
import torch 
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, confusion_matrix 
import seaborn as sns 

CLASS_NAMES = [
    'akiec (Actinic Keratoses)',
    'bcc (Basal Cell Carcinoma)',
    'bkl (Benign Keratosis)',
    'df (Dermatofibroma)',
    'mel (Melanoma)',
    'nv (Melanocytic Nevi)',
    'vasc (Vascular Lesions)'
]

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    print(f"[Utils] Random seed set to {seed}")
    
class TrainingLogger:
    def __init__(self, log_path: str = None):
        self.history = []
        self.log_path = log_path 
        
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok = True)
            
            with open(log_path, "w") as f:
                f.write("FL Training Log\n" + "="*60 + "\n\n")
                
    def log_round(self, round_num: int, client_results: list, test_metrics: dict) -> None:
        avg_train_loss = np.mean([r["train_loss"] for r in client_results])
        avg_train_acc = np.mean([r["train_acc"] for r in client_results])
        
        entry = {
            'round'          : round_num,
            'avg_train_loss' : avg_train_loss,
            'avg_train_acc'  : avg_train_acc,
            'test_loss'      : test_metrics.get('test_loss'),
            'test_acc'       : test_metrics.get('test_acc')
        }
        self.history.append(entry)
        
        log_line = (
            f"Round {round_num:3d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {avg_train_acc:.2f}% | "
            f"Test Loss: {test_metrics.get('test_loss', 'N/A'):.4f} | "
            f"Test Acc: {test_metrics.get('test_acc', 'N/A'):.2f}%"
            if test_metrics.get('test_acc') is not None
            else
            f"Round {round_num:3d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {avg_train_acc:.2f}%"
        )
        print(f"\n{'='*70}")
        print(f"  ROUND {round_num} SUMMARY")
        print(f"  {log_line}")
        print(f"{'='*70}\n")
        
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(log_line + "\n") 
    
    def save_history(self, json_path: str) -> None:
        with open(json_path, "w") as f:
            json.dump(self.history, f, indent = 2)
        print(f"[Logger] Training history saved -> {json_path}")
        
def plot_training_curves(history: list, save_path: str = None) -> None:
    rounds     = [entry['round']           for entry in history]
    train_loss = [entry['avg_train_loss']  for entry in history]
    train_acc  = [entry['avg_train_acc']   for entry in history]
    test_loss  = [entry.get('test_loss')   for entry in history]
    test_acc   = [entry.get('test_acc')    for entry in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Federated Learning Training Curves (HAM10000)', fontsize=14, fontweight='bold')

    axes[0].plot(rounds, train_loss, 'b-o', label='Avg Client Train Loss', markersize=4)
    if any(v is not None for v in test_loss):
        clean_rounds    = [r for r, v in zip(rounds, test_loss) if v is not None]
        clean_test_loss = [v for v in test_loss if v is not None]
        axes[0].plot(clean_rounds, clean_test_loss, 'r-s', label='Global Test Loss', markersize=4)
    axes[0].set_xlabel('FL Round')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss per Round')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, train_acc, 'b-o', label='Avg Client Train Acc', markersize=4)
    if any(v is not None for v in test_acc):
        clean_rounds   = [r for r, v in zip(rounds, test_acc) if v is not None]
        clean_test_acc = [v for v in test_acc if v is not None]
        axes[1].plot(clean_rounds, clean_test_acc, 'r-s', label='Global Test Acc', markersize=4)
    axes[1].set_xlabel('FL Round')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy per Round')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Utils] Training curves saved → {save_path}")
    plt.show()
    
def plot_confusion_matrix(y_true: list, y_pred: list, save_path: str = None) -> None:
    cm = confusion_matrix(y_true, y_pred)
    short_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix — Global Model on Test Set')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Utils] Confusion matrix saved → {save_path}")
    plt.show()

def print_classification_report(y_true: list, y_pred: list) -> None:
    short_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    report = classification_report(y_true, y_pred, target_names=short_names, zero_division = 0)
    print("\nPer-Class Classification Report:")
    print("="*60)
    print(report)

