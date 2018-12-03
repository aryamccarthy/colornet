"""Because we believe in good scholarship.
"""
import argparse
from collections import Counter
from pathlib import Path
from pprint import pprint
import sys; sys.path.append("..")
from typing import Dict, List
import pickle 
import subprocess
import os

from typing import Dict, List, Tuple
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from tensorboardX import SummaryWriter
import torch as th
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from data import DataLoader
from network import ColorNet
from data.dataset import ColorDataset
from data.vocab import ExtensibleVocab
import numpy as np

def euclidean_distance(a: th.Tensor, b: th.Tensor):
    assert len(a) == len(b)
    return th.sqrt(th.sum((a - b) ** 2))

def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def collate(batch):
    return batch

def get_data_loaders(train_batch_size: int, val_batch_size: int, device: int = None):
 #   training_dataset = ColorDataset("train")
#    training_generator = DataLoader(training_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate)

    training_generator = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "train", device=device)
    dev_generator = DataLoader("../../data/raw/xkcd_colordata", "../../data/raw/", "dev", device=device)
    #dev_dataset = ColorDataset("dev")
 #   dev_generator = DataLoader(dev_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate)

    return training_generator, dev_generator

def prepare_batch(batch, device, non_blocking=False):
    """Jump through hoops to work with `ignite`.

    `ignite` expects (x, y) pairs, but our network takes in dicts and 
    returns dicts as well. We create a useless `y` just to satisfy the API.
    """
    x = batch[0:-1]
    y = batch[-1]    
    if device is not None:
        x = [ins.cuda(device) for ins in x]
        y = y.cuda(device)
    return x, y

def loss_fn(y_pred: Tuple[th.Tensor], y: th.Tensor):
    """Jump through hoops to work with `ignite`.

    We ignore `y` (as with `prepare_batch`) and return only the relevant
    dictionary entry of `y_pred`. 
    """
    pred, reference, beta = y_pred
    cosine_fn = nn.CosineSimilarity(dim=0) 
    cosine_sim = cosine_fn(pred, y- reference) 
    euc_dist = euclidean_distance(reference + pred, y)
    loss1, loss2 = -cosine_sim, euc_dist
  
    agg_loss = loss1 + beta * loss2
    total_loss = th.mean(agg_loss, dim=0)
    return total_loss

def distance(y_pred: Tuple[th.Tensor], y: None):
    pred,reference,beta = y_pred
    euc_dist = euclidean_distance(reference + pred, y)
    return th.mean(euc_dist, dim=0)
    #return sum(output["distance"] for output in y_pred) / len(y_pred)

def angle(y_pred: Tuple[th.Tensor], y: None):
    pred,reference,beta = y_pred
    cosine_sim = nn.CosineSimilarity(dim=0)(pred, y - reference)
    return th.mean(cosine_sim, dim=0)
    #return sum(output["cosine_sim"] for output in y_pred) / len(y_pred)

def run(
        train_batch_size: int,
        val_batch_size: int,
        epochs: int,
        lr: float,
        log_interval: int,
        log_dir: Path,
        str_to_ids: Path,
        ids_to_str: Path,
        beta: float,
        device: int = None
        ) -> None:
    if device is not None:
        device = th.device(device)
    
    # init checkpointing 
    model_dir = os.path.join(log_dir, "models")
    try:
        os.mkdir(model_dir)
    except FileExistsError:
        pass
     
    # load vocab from pickled dict
    with open(str_to_ids, "rb") as f1, open(ids_to_str, "rb") as f2:
        str_to_ids = pickle.load(f1)
        ids_to_str = pickle.load(f2)
    # get the corresponding vectors, sorted by vocab index
    words = str_to_ids.keys() 
    freqs = Counter(words)
    vocab = ExtensibleVocab(freqs, vectors='fasttext.simple.300d')
    sorted_vocab_items = sorted(str_to_ids.items(), key=lambda x: x[1])
    num_embeddings, embedding_dim = len(words), list(vocab["<PAD>"].size())[0]
    # init empty array for embeddings
    print(num_embeddings, embedding_dim)
    embedding_arr = np.zeros((num_embeddings, embedding_dim))
    
    # fill the array in order
    for word, idx in sorted_vocab_items:
        corresponding_embedding = vocab[word]
        embedding_arr[idx,:] = corresponding_embedding 
    print("Getting train/dev loaders...")
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size, device=device)
    print("Got loaders")
    print("Defining model...")
    model = ColorNet(color_dim=3, vocab=str_to_ids, pretrained_embeddings = embedding_arr, beta=beta, device=device)
    print("Defined model")
    writer = create_summary_writer(model, train_loader, log_dir)

    handler = ModelCheckpoint(model_dir, f"{lr}-{beta}-{train_batch_size}", save_interval = 2, n_saved = 10, create_dir=True, require_empty=False)
    optimizer = Adam(model.parameters(), lr=lr)
    trainer = create_supervised_trainer(model, optimizer, loss_fn=loss_fn, prepare_batch=prepare_batch, device=device)
    # add checkpointing 
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler,  {"mymodel": model })
    evaluator = create_supervised_evaluator(model, 
                                            prepare_batch=prepare_batch, 
                                            device=device,
                                            metrics={'loss': Loss(loss_fn),
                                            'angle': Loss(angle),
                                            'distance': Loss(distance),
                                            })

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=train_loader.length,
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % train_loader.length + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

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
        writer.add_scalar(f"{split.lower()}/avg_loss", avg_loss, engine.state.epoch)
        writer.add_scalar(f"{split.lower()}/avg_angle", avg_angle, engine.state.epoch)
        writer.add_scalar(f"{split.lower()}/avg_distance", avg_distance, engine.state.epoch)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        log_results(engine, train_loader, "Training")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        log_results(engine, val_loader, "Validation")
        pbar.n = pbar.last_print_n = 0

    @trainer.on(Events.ITERATION_COMPLETED)
    def newline(engine):
        # format lines for tqdm
        print("") 


    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()
    writer.close()


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
    parser.add_argument('--log_interval', type=int, default=1,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=Path, default="../../models/tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--embedding_file", type=Path, default="../../data/embeddings/subset-wb.p",
                        help="Where the embeddings are stored")
    parser.add_argument("--beta", type=float, default=0.3/0.7,
                        help="Weight between objectives: loss = (-cosine_sim) + β * distance")
    parser.add_argument("--use-gpu", type=bool, default=False)
    args = parser.parse_args()
    return args

def main():
    args: argparse.Namespace = parse_args()
    if args.use_gpu:
        try:
            output = subprocess.check_output("free-gpu", shell=True)
            output=int(output.decode('utf-8'))
            gpu = output
            print("claiming gpu {}".format(gpu))
            th.cuda.set_device(int(gpu))
            a = th.tensor([1]).cuda()
            device = int(gpu)
        except (IndexError, subprocess.CalledProcessError) as e:
            device = None
    else:
        device = None

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.log_interval, args.log_dir, "../../data/embeddings/str_to_ids.pkl", "../../data/embeddings/ids_to_str.pkl", args.beta, device)

if __name__ == '__main__':
    main()
