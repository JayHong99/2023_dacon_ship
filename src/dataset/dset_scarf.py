from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
import random

class dset_Wrapper :
    def __init__(self, fold_dir : Path, drop_cols : list, target_col : str) :
        self.fold_dir = fold_dir
        self.fold_data_dir = fold_dir / 'data'
        self.drop_cols = drop_cols
        self.target_col = target_col
            
    def load_dset(self, mode : str) : 
        df = pd.read_csv(self.fold_data_dir / f'{mode}.csv')
        target = df[self.target_col] if mode != 'test' else np.zeros(len(df))
        df = df.drop(self.drop_cols, axis = 1) if mode == 'test' else df.drop(self.drop_cols + [self.target_col], axis = 1)
        self.input_dim = len(df.columns)
        
        dset = customDatset(df, target)
        return dset
    
    def load_dataloaders(self, batch_size, num_workers, need_train_shuffle = True) : 
        self.train_dset = self.load_dset('train')
        self.valid_dset = self.load_dset('valid')
        self.test_dset  = self.load_dset('test')
        return {
            'train' : DataLoader(self.train_dset, batch_size = batch_size, shuffle = need_train_shuffle, num_workers = num_workers, pin_memory = True, persistent_workers=True),
            'valid' : DataLoader(self.valid_dset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = True, persistent_workers=True),
            'test'  : DataLoader(self.test_dset,  batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = True, persistent_workers=True),
        }

    

class customDatset(Dataset) : 
    def __init__(self, df : pd.DataFrame, target : np.array) :
        self.df = df.values
        self.target = target
        
    def __getitem__(self, index : int) :
        random_idx = random.randint(0, len(self)-1)
        random_sample = torch.tensor(self.df[random_idx], dtype = torch.float)
        sample = torch.tensor(self.df[index], dtype = torch.float)
        return random_sample, sample, self.target[index]

    def __len__(self) :
        return len(self.target)