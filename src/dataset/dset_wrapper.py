from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
import random

class dset_Wrapper :
    def __init__(self, fold_dir : Path, cat_cols : list, num_cols : list, drop_cols : list, target_col : str, corrupt_rate : float) :
        self.fold_dir = fold_dir
        self.fold_data_dir = fold_dir / 'data'
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.drop_cols = drop_cols
        self.target_col = target_col
        self.corrupt_rate = corrupt_rate
            
    def load_dset(self, mode : str) : 
        df = pd.read_csv(self.fold_data_dir / f'{mode}.csv')
        cat_df = df[self.cat_cols]
        con_df = df[self.num_cols]
        target = df[self.target_col] if mode != 'test' else np.zeros(len(df))
        
        if mode =='train' :        
            dset = customDatset(cat_df, con_df, target, mode, self.corrupt_rate, None, None)
        else :
            dset = customDatset(cat_df, con_df, target, mode, self.corrupt_rate, self.cat_distribution, self.con_distribution)
        return dset
        
    
    def load_dataloaders(self, batch_size, num_workers, need_train_shuffle = True) : 
        self.train_dset = self.load_dset('train')
        self.set_train_information()
        self.valid_dset = self.load_dset('valid')
        self.test_dset  = self.load_dset('test')
        return {
            'train' : DataLoader(self.train_dset, batch_size = batch_size, shuffle = need_train_shuffle, num_workers = num_workers, pin_memory = True, persistent_workers=True),
            'valid' : DataLoader(self.valid_dset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = True, persistent_workers=True),
            'test'  : DataLoader(self.test_dset,  batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = True, persistent_workers=True),
        }
    
    def set_train_information(self) : 
        self.cat_distribution, self.con_distribution = self.train_dset.cat_distribution, self.train_dset.con_distribution
        self.cat_unique_dims = list(self.train_dset.cat_df.nunique().values + 2) # +2 for unknown and padding
        self.con_features_dims = self.train_dset.con_df.shape[1]


    

class customDatset(Dataset) : 
    def __init__(self, cat_df, con_df, target, mode : str, corrupt_rate : float, cat_distribution : list, con_distribution : list) :
        self.cat_df = cat_df
        self.con_df = con_df
        self.target = target
        self.mode = mode
        self.c = corrupt_rate
        if mode == 'train' : 
            self.cat_distribution = cat_df.transpose().values.tolist()
            self.con_distribution = con_df.transpose().values.tolist()
        else : 
            self.cat_distribution = cat_distribution
            self.con_distribution = con_distribution
    
    def corrupt(self, x, con_or_cat : str) : 
        x = copy.deepcopy(x)
        indices = random.sample(range(len(x)), int(len(x) * self.c))
        for i in indices : 
            x[i] = random.choice(self.cat_distribution[i]) if con_or_cat == 'cat' else random.choice(self.con_distribution[i])
        return x

    def __getitem__(self, index : int) :
        cat_x = self.cat_df.iloc[index].values.tolist()
        cat_origin = torch.tensor(cat_x, dtype = torch.long)
        cat_corrupt = torch.tensor(self.corrupt(cat_x, 'cat'), dtype = torch.long)
        con_x = self.con_df.iloc[index].values.tolist()
        con_origin = torch.tensor(con_x, dtype = torch.float)
        con_corrupt = torch.tensor(self.corrupt(con_x, 'con'), dtype = torch.float)
        label = self.target[index]
        return cat_origin, cat_corrupt, con_origin, con_corrupt, label

    def __len__(self) :
        return len(self.target)