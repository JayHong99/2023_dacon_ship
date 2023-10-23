
from typing import Tuple, List

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from .loss import NTXent

def pick_gpu_lowest_memory():
    import gpustat
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.memory_used)/float(gpu.memory_total), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU

def train_epoch(model, optimizer, train_loader, device) : 
    model.train()
    linreg = LinearRegression()
    # criterion = CLIPLoss(temperature=0.1)
    criterion = NTXent(temperature=0.1)
    total_loss, total_mae = 0, 0
    for X_origin, X_random, y_num in train_loader: 
        X_origin = X_origin.to(device, non_blocking=True)
        X_random = X_random.to(device, non_blocking=True)
        y_num = y_num.to(device, non_blocking=True)
        
        emb, emb_corruptted = model(X_origin, X_random)
        
        loss = criterion(emb, emb_corruptted)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        linreg, simple_mae = simple_acc_with_logreg(emb, y_num, linreg, need_fit = True)
        total_loss += loss.item()
        total_mae += simple_mae

    total_loss = total_loss / len(train_loader)
    total_mae = total_mae / len(train_loader)
    return total_loss, total_mae, linreg

def evaluate_epoch(model, eval_loader, device, linreg = None) :
    model.eval()
    # criterion = CLIPLoss(temperature=0.1)
    criterion = NTXent(temperature=0.1)
    total_loss, total_mae = 0, 0
    with torch.no_grad() : 
        for X_origin, X_random, y_num in eval_loader: 
            X_origin = X_origin.to(device, non_blocking=True)
            X_random = X_random.to(device, non_blocking=True)
            y_num = y_num.to(device, non_blocking=True)
            
            emb, emb_corruptted = model(X_origin, X_random)
            
            loss = criterion(emb, emb_corruptted)
            
            _, simple_mae = simple_acc_with_logreg(emb, y_num, linreg) if linreg is not None else (None, 0)
            total_loss += loss.item()
            total_mae += simple_mae
    total_loss = total_loss / len(eval_loader)
    total_mae = total_mae / len(eval_loader)
    return total_loss, total_mae

def extract_feature(model, eval_laoder, device) : 
    model.eval()
    embs = []
    labels = []
    with torch.no_grad() : 
        for X_cat_origin, _, X_con_origin, _, y_num in eval_laoder :
            X_cat_origin = X_cat_origin.to(device, non_blocking=True)
            X_con_origin = X_con_origin.to(device, non_blocking=True)
            _, emb = model(X_cat_origin, X_con_origin)
            embs.append(emb.cpu().detach().numpy())
            labels.append(y_num.cpu().detach().numpy())
    return np.concatenate(embs), np.concatenate(labels)
    

def simple_acc_with_logreg(emb, label, linreg, need_fit = False) : 
    emb = emb.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    if need_fit : 
        linreg.fit(emb, label)
    pred = linreg.predict(emb)
    simple_mae = np.mean(np.abs(pred - label))
    return linreg, simple_mae

class CLIPLoss(torch.nn.Module):
  """
  Loss function for multimodal contrastive learning based off of the CLIP paper.
  
  Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
  similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
  Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal. 
  """
  def __init__(self, 
               temperature: float,
               lambda_0: float = 0.5) -> None:
    super(CLIPLoss, self).__init__()

    self.temperature = temperature
    self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    if lambda_0 > 1 or lambda_0 < 0:
      raise ValueError('lambda_0 must be a float between 0 and 1.')
    self.lambda_0 = lambda_0
    self.lambda_1 = 1-lambda_0

  def forward(self, out0: torch.Tensor, out1: torch.Tensor, indices: List[int] = None) -> Tuple:
    # normalize the embedding onto the unit hypersphere
    device = out0.device
    out0 = nn.functional.normalize(out0, dim=1)
    out1 = nn.functional.normalize(out1, dim=1)

    #logits = torch.matmul(out0, out1.T) * torch.exp(torch.tensor(self.temperature))
    logits = torch.matmul(out0, out1.T) / self.temperature
    labels = torch.arange(len(out0), device=device)
    
    loss_0 = self.lambda_0 * self.cross_entropy(logits, labels)
    loss_1 = self.lambda_1 * self.cross_entropy(logits.T, labels)
    loss = loss_0 + loss_1
    return loss, logits, labels