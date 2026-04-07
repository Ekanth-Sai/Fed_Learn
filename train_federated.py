import os
import argparse
import time 

from src.server import FederatedServer 
from src.client import FederatedClient
from src.utils import set_seed, TrainingLogger, plot_training_curves 

def build_clients(args, device: str) -> list:
    clients = []
    for i in range(args.n_clients):
        csv_path = os.path.join(args.client_data_dir, f"client_{i}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Client {i} CSV not found at {csv_path}.\n"
                f"Run partition_data.py first"
            )
        client = FederatedClient(
            client_id = i,
            csv_path = csv_path,
            num_classes = 7,
            local_epochs = args.local_epochs,
            batch_size = args.batch_size,
            lr = args.lr,
            # lr = args.lr / (1 + 0.1 * round_num),
            device = device
        )
        clients.append(client)
    
    return clients

def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok = True)
    
    device = "cpu"
    print(f"\n[Setup] Using device: {device}")
    print(f"[Setup] FL config: {args.n_clients} clients, "
            f"{args.num_rounds} rounds, "
            f"{args.local_epochs} local epochs/round")
    
    test_csv = os.path.join(args.client_data_dir, "test.csv")
    test_csv = test_csv if os.path.exists(test_csv) else None 
    
    print("\n[Init] Building server...")
    server = FederatedServer(
        num_classes = 7,
        test_csv = test_csv,
        device = device
    )
    
    print("\n[Init] Building clients...")
    clients = build_clients(args, device)
    
    log_path = os.path.join(args.output_dir, "training_log.txt")
    logger = TrainingLogger(log_path = log_path)
    
    print(f"\n{'#'*70}")
    print(f"STARTING FEDERATED LEARNING TRAINING")
    print(f"Rounds: {args.num_rounds} | Clients: {args.n_clients}")
    print(f"{'#'*70}\n")
    
    best_test_acc = 0.0
    
    for round_num in range(1, args.num_rounds + 1):
        round_start = time.time()
        
        print(f"\n{'-'*70}")
        print(f"ROUND {round_num}/{args.num_rounds}")
        print(f"{'-'*70}")

        global_params = server.get_global_parameters()
        
        client_results = []
        current_lr = args.lr / (1 + 0.1 * round_num)
        print(f"Learning rate this round: {current_lr:.6f}")
        
        for client in clients:
            print(f"\n -> Training Client {client.client_id}...")
            result = client.train_one_round(global_params, current_lr)
            client_results.append(result)
            print(f"Client {client.client_id} done | "
                    f"Loss: {result['train_loss']:.4f} | "
                    f"Acc: {result['train_acc']:.2f}%")
        
        print(f"\n -> Server aggregating with FedAvg...")
        server.aggregate(client_results)
        
        print(f" -> Evaluating global model...")
        test_metrics = server.evaluate() 
        
        if test_metrics['test_acc'] is not None:
            print(f" -> Global Test Loss: {test_metrics['test_loss']:.4f} | "
                    f"Test Acc: {test_metrics['test_acc']:.2f}%")
            
            if test_metrics['test_acc'] > best_test_acc:
                best_test_acc = test_metrics['test_acc']
                best_model_path = os.path.join(args.output_dir, 'best_global_model.pth')
                server.save_model(best_model_path)
                print(f" -> New best model! Acc: {best_test_acc:.2f}%")
        
        logger.log_round(round_num, client_results, test_metrics)
        
        round_time = time.time() - round_start
        print(f"\n Round {round_num} completed in {round_time:.1f} seconds.")
        
    print(f"\n{'#'*70}")
    print(f"TRAINING COMPLETE")
    
    if best_test_acc > 0:
        print(f"Best Global Test Accuracy: {best_test_acc:.2f}%")
    print(f"{'#'*70}\n")
    
    final_model_path = os.path.join(args.output_dir, "final_global_model.pth")
    server.save_model(final_model_path) 
    
    history_path = os.path.join(args.output_dir, 'training_history.json')
    logger.save_history(history_path) 
    
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_curves(logger.history, save_path = curves_path)
    
    print(f"\nOutputs saved to: {args.output_dir}/")
    print(f"  - final_global_model.pth")
    print(f"  - best_global_model.pth")
    print(f"  - training_log.txt")
    print(f"  - training_history.json")
    print(f"  - training_curves.png")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Federated Learning Training on HAM10000"
    )
    parser.add_argument(
        "--client_data_dir",
        type = str,
        default = "client_data",
        help = "Directory with client_0.csv ... client_N.csv and test.csv"
    )
    parser.add_argument(
        "--n_clients",
        type = int,
        default = 5,
        help = "Number of clients (must match the partitioned data)"
    )
    parser.add_argument(
        "--num_rounds",
        type = int,
        default = 10,
        help = "Number of FL communication rounds"
    )
    parser.add_argument(
        "--local_epochs",
        type = int,
        default = 2,
        help = "Local training epochs per round per client"
    )
    parser.add_argument(
        "--batch_size",
        type = int,
        default = 16,
        help = "Training batch size per client"
    )
    parser.add_argument(
        "--lr",
        type = float,
        default = 0.01,
        help = "Learning rate for local SGD optimizer"
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "outputs",
        help = "Directory to save models, logs, and plots"
    )
    parser.add_argument(
        '--seed',
        type = int,
        default = 42,
        help = "Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    main(args)