import os 
import argparse
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split

def find_image_path(image_id: str, image_dirs: list) -> str:
    for directory in image_dirs:
        path = os.path.join(directory, image_id + '.jpg')
        
        if os.path.exists(path):
            return path
        
    raise FileNotFoundError(
        f"Image '{image_id}.jpg' not found in any of: {image_dirs}"
    )

def dirichlet_partition(df: pd.DataFrame, n_clients: int, alpha: float, seed: int) -> list:
    np.random.seed(seed)
    classes = df['dx'].unique()
    client_indices = [[] for _ in range(n_clients)]
    
    for cls in classes:
        cls_indices = df[df['dx'] == cls].index.tolist() 
        np.random.shuffle(cls_indices)

        proportions = np.random.dirichlet(alpha=[alpha] * n_clients)

        counts = (proportions * len(cls_indices)).astype(int) 
        
        remainder = len(cls_indices) - counts.sum()
        counts[-1] += remainder  
        
        start = 0
        for i, count in enumerate(counts):
            end = start + count
            client_indices[i].extend(cls_indices[start:end])
            start = end
    
    client_dfs = [df.loc[indices].reset_index(drop = True) for indices in client_indices]
    return client_dfs 

def plot_client_distribution(client_dfs: list, alpha: float, save_path: str = None) -> None:
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    n_clients = len(client_dfs)
    
    fig, axes = plt.subplots(1, n_clients, figsize = (4 * n_clients, 5), sharey = False)
    if n_clients == 1:
        axes = [axes]
        
    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    
    for i, (client_df, ax) in enumerate(zip(client_dfs, axes)):
        counts = client_df['dx'].value_counts().reindex(classes, fill_value = 0)
        bars = ax.bar(classes, counts.values, color = colors, edgecolor = 'black', linewidth = 0.5)
        ax.set_title(f"Client {i}\n(n = {len(client_df)})", fontsize = 10)
        ax.set_xlabel("Class", fontsize = 8)
        ax.set_ylabel("Sample Count", fontsize = 8)
        ax.tick_params(axis = "x", rotation = 45, labelsize = 7)
        ax.grid(axis = 'y', alpha = 0.3)

    plt.suptitle(f"Non-IID Data Distribution Across Clients (Dirichlet = {args.alpha})", fontsize = 12, fontweight = "bold", y = 1.02)
    plt.tight_layout() 
    
    if save_path:
        plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
        print(f"[Partition] Distribution plot saved -> {save_path}")
    plt.show()
    
def main(args):
    print("\n" + "=" * 60)
    print("HAM10000 Non-IID Data Partitioning")
    print("="*60)
    
    print(f"\n[1/5] Loading metadata from: {args.metadata}")
    df = pd.read_csv(args.metadata)
    print(f"Total records: {len(df)}")
    print(f"Class distribution:\n{df['dx'].value_counts().to_string()}")
    
    print(f"\n[2/5] Resolving image paths from: {args.image_dirs}")
    image_dirs = args.image_dirs 
    
    resolved_paths = []
    missing = 0
    
    for _, row in df.iterrows():
        try:
            path = find_image_path(row['image_id'], image_dirs)
            resolved_paths.append(path)
        except FileNotFoundError:
            resolved_paths.append(None)
            missing += 1
            
    df['image_path'] = resolved_paths
    df = df[df['image_path'].notna()].reset_index(drop = True)
    print(f"Resolved: {len(df)} images | Missing/skipped: {missing}")
    
    print(f"\n[3/5] Splitting into train ({1-args.test_ratio:.0%}) / "
            f"test ({args.test_ratio:.0%})")
    
    train_df, test_df = train_test_split(
        df,
        test_size = args.test_ratio,
        stratify = df['dx'],
        random_state = args.seed
    )
    
    train_df = train_df.reset_index(drop = True)
    test_df = test_df.reset_index(drop = True)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    
    print(f"\n[4/5] Partitioning into {args.n_clients} clients "
            f"(Dirichlet = {args.alpha})")
    
    client_dfs = dirichlet_partition(
        df = train_df,
        n_clients = args.n_clients,
        alpha = args.alpha,
        seed = args.seed
    )
    
    min_size = min(len(cdf) for cdf in client_dfs)
    client_dfs = [cdf.sample(n=min_size, random_state=args.seed).reset_index(drop=True)
              for cdf in client_dfs]
    print(f"[Balance] All clients downsampled to {min_size} samples each")
    
    for i, cdf in enumerate(client_dfs):
        print(f"Client {i}: {len(cdf)} samples | "
                f"Classes: {dict(cdf['dx'].value_counts())}")
        
    print(f"\n[5/5] Saving client CSVs to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok = True)
    
    for i, cdf in enumerate(client_dfs):
        path = os.path.join(args.output_dir, f"client_{i}.csv")
        cdf[["image_path", "dx", "image_id"]].to_csv(path, index = False)
        print(f"Saved: {path}")
    
    test_path = os.path.join(args.output_dir, "test.csv")
    test_df[["image_path", "dx", "image_id"]].to_csv(test_path, index = False)
    print(f"Saved test set: {test_path}")
    
    os.makedirs('outputs', exist_ok = True)
    plot_client_distribution(client_dfs, alpha = args.alpha, save_path = 'outputs/client_distribution.png')
    
    print("\nPartitioning complete")
    print(f"Client CSVs: {args.output_dir}/client_0.csv ... client_{args.n_clients - 1}.csv")
    print(f"Test CSV: {args.output_dir}/test.csv")
    print("="*60 + "\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'Partition HAM10000 into Non-IID federated client datasets'
    ) 
    
    parser.add_argument(
        "--metadata",
        type = str,
        required = True,
        help = "Path to HAM10000_metadata.csv"
    )
    
    parser.add_argument(
        "--image_dirs",
        type = str,
        nargs = "+",
        required = True,
        help = "One or more directories containing .jpg images"
    )
    
    parser.add_argument(
        "--output_dir",
        type = str,
        default = "client_data",
        help = "Directory where client CSV files will be saved"
    )
    
    parser.add_argument(
        "--n_clients",
        type = int,
        default = 5,
        help = "Number of federated clients (default: 5)"
    )
    
    parser.add_argument(
        "--alpha",
        type = float,
        default = 0.5,
        help = "Dirichlet alpha: smaller = more Non-IID (default: 0.5)"
    )
    
    parser.add_argument(
        "--test_ratio",
        type = float,
        default = 0.15,
        help = "Fraction of data for global test set (default: 0.15)"
    )
    
    parser.add_argument(
        "--seed",
        type = int,
        default = 42,
        help = "Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    main(args)