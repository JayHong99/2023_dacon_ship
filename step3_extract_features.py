from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import os
import pickle

from src.models.tab_transformer_embedder import TabTransformer_Embedder
from src.dataset.dset_wrapper import dset_Wrapper
from src.utils import pick_gpu_lowest_memory, extract_feature


root_dir = Path('.')
data_dir = root_dir / 'data'
result_dir = root_dir / 'Results'

## Hyperparameters
corrupt_ratio = 0.50
embedding_dim = 128
projection_dim = 128
batch_size = 4096
num_workers = 4

fold_num = int(sys.argv[1])

fold_dir = result_dir / f'fold_{fold_num}'
device = pick_gpu_lowest_memory()
model_dir = fold_dir.joinpath('Model')
emb_save_path = fold_dir.joinpath('data', 'result_emb.pkl')

# device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
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
    dataloaders = data_wrapper.load_dataloaders(batch_size, num_workers, need_train_shuffle=False)
    
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
        )

    model_paths = list(model_dir.rglob('*.pt'))
    model_paths = sorted(model_paths, key = lambda x : int(x.stem.split('_')[-1]))
    # model_path = model_paths[-1]
    model_path = Path('Results/fold_0/Model/model_36.pt')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    train_loader, valid_loader, test_loader = dataloaders.get('train'), dataloaders.get('valid'), dataloaders.get('test')
    train_embeddings, train_labels = extract_feature(model, train_loader, device)
    print(f"TRAIN : {train_embeddings.shape}, {train_labels.shape}")
    valid_embeddings, valid_labels = extract_feature(model, valid_loader, device)
    print(f"VALID : {valid_embeddings.shape}, {valid_labels.shape}")
    test_embeddings, test_labels = extract_feature(model, test_loader, device)
    print(f"TEST : {test_embeddings.shape}, {test_labels.shape}")
    
    
    with open(emb_save_path, 'wb') as f:
        pickle.dump([train_embeddings, train_labels, valid_embeddings, valid_labels, test_embeddings, test_labels], f)
    
    
if __name__ == '__main__':
    main()