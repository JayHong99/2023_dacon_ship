from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import sys

from src.models.model import SCARF
# from src.dataset.dset_wrapper import dset_Wrapper
from src.dataset.dset_scarf import dset_Wrapper
from src.utils import pick_gpu_lowest_memory, train_epoch, evaluate_epoch
from src.logger import print_logger

root_dir = Path('.')
data_dir = root_dir / 'data'
result_dir = root_dir / 'Results'

## Hyperparameters
corrupt_ratio = 0.50
embedding_dim = 128
batch_size = 1024
num_workers = 4
num_epochs = 100
initial_lr = 0.1


fold_num = int(sys.argv[1])
fold_dir = result_dir / f'fold_{fold_num}'
device = pick_gpu_lowest_memory()
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
device = "cuda"

drop_cols = [
    'SAMPLE_ID', 
    "ID", # 선박 ID
    "SHIPMANAGER", # 선박 소유주
    # 'CI_HOUR', 
    ]

def main():
    data_wrapper = dset_Wrapper(fold_dir, drop_cols, 'CI_HOUR')
    dataloaders = data_wrapper.load_dataloaders(batch_size, num_workers)
    
    model = SCARF(
                    input_dim = data_wrapper.input_dim,
                    emb_dim=embedding_dim,
                    corruption_rate=corrupt_ratio,
                ).to(device)

    model_save_dir = fold_dir.joinpath('Model')
    model_save_dir.parent.mkdir(parents = True, exist_ok = True)
    
    logger = print_logger(fold_dir, 'training_log.txt')
    
    optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs, eta_min = 1e-5)
    
    
    for epoch in range(num_epochs) : 
        lr = optimizer.param_groups[0]['lr']
        total_loss, total_mae, linreg = train_epoch(model, optimizer, dataloaders['train'], device)
        logger(f'TRAIN | Epoch {epoch} | Loss : {total_loss:.4f} | MAE : {total_mae:.4f} | LR : {lr:.6f}')
        
        total_loss, total_mae = evaluate_epoch(model, dataloaders['valid'], device, linreg)
        logger(f"VALID | Epoch {epoch} | Loss : {total_loss:.4f} | MAE : {total_mae:.4f} | LR : {lr:.6f}")
        scheduler.step()
        
        model_weight = model.state_dict()
        model_save_path = model_save_dir.joinpath(f'model_{epoch}.pt')
        torch.save(model_weight, model_save_path)
        
if __name__ == '__main__':
    main()
    