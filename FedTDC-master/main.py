# main.py
import argparse
import numpy as np
import torch

from server import FedTDCServer
from client import FedTDCClient
from dataset.dataset import generate_dataset


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")

    # Training parameters
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--local_epochs", type=int, default=5, help="Number of local training epochs")
    parser.add_argument("--global_epochs", type=int, default=50, help="Maximum number of global rounds")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluation interval")

    # Federated learning parameters
    parser.add_argument("--join_ratio", type=float, default=1.0, help="Ratio of clients participating in each round")
    parser.add_argument("--random_join_ratio", type=bool, default=False, help="Whether to use random join ratio")

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='MNIST',
                        choices=['MNIST', 'Cifar10', 'Cifar100', 'FashionMNIST', 'OfficeCaltech10', 'DomainNet'])
    parser.add_argument('--base_data_dir', type=str, default='/data/HYQ/FedTDC/data', help='Base directory for datasets')
    parser.add_argument('--num_clients', type=int, default=20, help='Number of clients')
    parser.add_argument('--noniid', type=bool, default=True, help='Whether to use non-IID data distribution')
    parser.add_argument('--balance', type=bool, default=False, help='Whether to use balanced data distribution')
    parser.add_argument('--partition', type=str, default='dir', choices=['pat', 'dir', 'exdir'])
    parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet distribution parameter')
    parser.add_argument('--train_ratio', type=float, default=0.75, help='Train/test split ratio')

    parser.add_argument("--dp_sigma", type=float, default=0.0, help="Gaussian noise std for DP on gf_mean")
    parser.add_argument('--use_dp', action='store_true', help='Enable differential privacy')
    parser.add_argument('--adaptive_dp', action='store_true', help='Enable adaptive DP mechanism')

    # Model params
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden layer size (generator/decoder)")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--pool_h", type=int, default=4, help="Adaptive pool height")
    parser.add_argument("--pool_w", type=int, default=4, help="Adaptive pool width")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # generate dataset partitions, returns experiment name
    experiment_name = generate_dataset(args)
    args.experiment_name = experiment_name

    # adjust dataset-specific defaults
    if args.dataset == "DomainNet":
        args.num_clients = 6
    elif args.dataset == "OfficeCaltech10":
        args.num_clients = 4

    # set num_classes per dataset
    if args.dataset in ["MNIST", "FashionMNIST"]:
        args.num_classes = 10
    elif args.dataset in ("Cifar10", "Cifar-10"):
        args.num_classes = 10
    elif args.dataset in ("Cifar100", "Cifar-100"):
        args.num_classes = 100
    elif args.dataset == "OfficeCaltech10":
        args.num_classes = 10
    elif args.dataset == "DomainNet":
        args.num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(args)

    best_accs = []
    print("Training starts...")

    for i in range(args.num_trials):
        print(f"Trial {i+1} of {args.num_trials}")
        # create clients
        clients = [FedTDCClient(args, client_id) for client_id in range(args.num_clients)]
        # create server with clients
        server = FedTDCServer(args, clients)
        server.run()
        best_accs.append(server.best_test_acc)

    print(f"test accuracy: {np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}")
    print(f"max test accuracy: {np.max(best_accs):.4f}")


if __name__ == "__main__":
    main()
