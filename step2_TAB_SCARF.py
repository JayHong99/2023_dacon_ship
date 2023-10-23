from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import sys

from src.models.tab_transformer_embedder import TabTransformer_Embedder
from src.dataset.dset_wrapper import dset_Wrapper
from src.utils import pick_gpu_lowest_memory, train_epoch, evaluate_epoch
from src.logger import print_logger

root_dir = Path('.')
data_dir = root_dir / 'data'
result_dir = root_dir / 'Results'

## Hyperparameters
corrupt_ratio = 0.50
embedding_dim = 128
projection_dim = 128
batch_size = 4096
num_workers = 4
num_epochs = 100
initial_lr = 0.1


fold_num = int(sys.argv[1])
fold_dir = result_dir / f'fold_{fold_num}'
device = pick_gpu_lowest_memory()
os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
device = "cuda"

str_cols = [ # Label encoding 변환을 위하여 선정
    "ARI_CO", # 도착항 소속 국가
    "ARI_PO", # 도착항 항구 명
    "SHIP_TYPE_CATEGORY",  # 5대 선종
    
    "FLAG", # 선박 국적
    "BREADTH", # 선박 폭
    "DEPTH", # 선박 깊이
    "DRAUGHT", # 선박 최대 허깨비
    "ATA_LT", # 도착항 시간F
    'BUILT', # 선박 나이
    
    ###################### 전처리로 추가한 변수들 ######################
    "year", # 연도
    "month", # 월
    "weekday", # 요일
    "morning", # 오전 여부
]

num_cols = [
    'DEADWEIGHT',
    'PORT_SIZE',
    'DIST',
    'GT', # 선박 총톤수
]

drop_cols = [
    'SAMPLE_ID', 
    "ID", # 선박 ID
    "SHIPMANAGER", # 선박 소유주
    # 'CI_HOUR', 
    ]

def main():
    data_wrapper = dset_Wrapper(fold_dir, str_cols, num_cols, drop_cols, 'CI_HOUR', corrupt_ratio)
    dataloaders = data_wrapper.load_dataloaders(batch_size, num_workers)
    
    cat_unique_dims = data_wrapper.cat_unique_dims
    con_features_num = data_wrapper.con_features_dims
    print(f'cat_unique_dims : {cat_unique_dims}, {len(str_cols)}')
    
    model = TabTransformer_Embedder(
            categories = cat_unique_dims,    
            num_continuous = con_features_num,
            dim = 64,                           # dimension, paper set at 32
            dim_out = embedding_dim,                        # binary prediction, but could be anything
            proj_dim = projection_dim,     
            depth = 12,                          # depth, paper recommended 6
            heads = 8,                          # heads, paper recommends 8
            attn_dropout = 0.1,                 # post-attention dropout
            ff_dropout = 0.05,                   # feed forward dropout
            mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
            mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            # continuous_mean_std = cont_mean_std # (optional) - normalize the continuous values before layer norm
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
    