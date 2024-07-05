import torch
from dataclasses import dataclass
from torch.optim import Adam

@dataclass
class FB15kConfig:
    epochs: int = 10
    eval_epoch: int = 10
    print_step: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir: str = 'FB15k'
    save_dir: str = 'model'
    save_epoch: int = 1
    batch_size: int = 128
    shuffle: bool = True
    optimizer: Adam = Adam
    lr: float = 0.001

@dataclass
class WN18Config:
    epochs: int = 50
    eval_epoch: int = 5
    print_step: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir: str = 'WN18'
    save_dir: str = 'model'
    save_epoch: int = 1
    batch_size: int = 128
    shuffle: bool = True
    optimizer: Adam = Adam
    lr: float = 0.001

