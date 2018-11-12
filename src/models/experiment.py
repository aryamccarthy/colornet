"""Because we believe in good scholarship.
"""
import argparse
from pathlib import Path

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from network import ColorNet

def run(
        train_batch_size: int,
        val_batch_size: int,
        epochs: int,
        lr: float,
        momentum: float,
        log_interval: int,
        log_dir: Path,
        embedding_file: Path,
        ) -> None:
    model = ColorNet(color_dim=3, embedding_dim=300, embedding_file=embedding_file)

    # TODO: Finish implementing based on https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist_with_tensorboardx.py
    # WAITING_FOR: data loaders that give actual data

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=Path, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--embedding_file", type=Path, default="../../data/external/wiki-news-300d-1M-subword.vec",
                        help="Where the embeddings are stored")

    args = parser.parse_args()
    return args

def main():
    args: argparse.Namespace = parse_args()
    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.log_interval, args.log_dir, args.embedding_file)

if __name__ == '__main__':
    main()