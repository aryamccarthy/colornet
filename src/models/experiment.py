"""Because we believe in good scholarship.
"""
import argparse
from pathlib import Path
from pprint import pprint
import sys; sys.path.append("..")
from typing import Dict, List

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from network import ColorNet
from data.dataset import ColorDataset

def collate(batch):
    return batch

def get_data_loaders(train_batch_size: int, val_batch_size: int):
    training_dataset = ColorDataset("train")
    training_generator = DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate)

    dev_dataset = ColorDataset("dev")
    dev_generator = DataLoader(dev_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate)

    return training_generator, dev_generator

def prepare_batch(batch, device=None, non_blocking=False):
    """Jump through hoops to work with `ignite`.

    `ignite` expects (x, y) pairs, but our network takes in dicts and 
    returns dicts as well. We create a useless `y` just to satisfy the API.
    """
    x = batch
    y = torch.Tensor([0.0 for _ in batch])
    return x, y

def loss_fn(y_pred: List[Dict], y: None):
    """Jump through hoops to work with `ignite`.

    We ignore `y` (as with `prepare_batch`) and return only the relevant
    dictionary entry of `y_pred`. 
    """
    total_loss = sum(output["loss"] for output in y_pred) / len(y_pred)
    return total_loss

def distance(y_pred: List[Dict], y: None):
    return sum(output["distance"] for output in y_pred) / len(y_pred)

def angle(y_pred: List[Dict], y: None):
    return sum(output["cosine_sim"] for output in y_pred) / len(y_pred)

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
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = ColorNet(color_dim=3, embedding_dim=300, embedding_file=embedding_file)

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(model, optimizer, loss_fn=loss_fn, prepare_batch=prepare_batch)
    evaluator = create_supervised_evaluator(model, prepare_batch=prepare_batch,
                                            metrics={'loss': Loss(loss_fn),
                                            'angle': Loss(angle),
                                            'distance': Loss(distance),
                                            })

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    def log_results(engine, loader: DataLoader, split: str) -> None:
        evaluator.run(loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['loss']
        avg_angle = metrics['angle']
        avg_distance = metrics['distance']
        tqdm.write(
            "{} Results - Epoch: {}  Avg loss: {:.2f} Avg angle: {:.2f} Avg distance: {:.2f}"
            .format(split, engine.state.epoch, avg_loss, avg_angle, avg_distance)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        log_results(engine, train_loader, "Training")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        log_results(engine, val_loader, "Validation")
        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=Path, default="../../models/tensorboard_logs",
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