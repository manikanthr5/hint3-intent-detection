import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

def train_fn(model, dataloader, loss_fn, optimizer, device):
    model.ffn.train()
    
    losses = []
    labels = []
    outputs = []
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_ = batch['label'].to(device)
        outputs_ = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs_, labels_)
        loss.backward()
        optimizer.step()
        
        labels.extend(labels_.detach().cpu().numpy())
        outputs.extend(torch.argmax(outputs_, axis=1).detach().cpu().numpy())
        losses.append(loss.item())
    
    loss = np.mean(losses)
    acc = accuracy_score(labels, outputs)
    f1 = f1_score(labels, outputs, average='weighted')
    
    return loss, acc, f1

def val_fn(model, dataloader, loss_fn, device):
    model.to(device)
    model.eval()
    
    losses = []
    labels = []
    outputs = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_ = batch['label'].to(device)
        outputs_ = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs_, labels_)
        
        labels.extend(labels_.detach().cpu().numpy())
        outputs.extend(torch.argmax(outputs_, axis=1).detach().cpu().numpy())
        losses.append(loss.item())
        
    loss = np.mean(losses)
    acc = accuracy_score(labels, outputs)
    f1 = f1_score(labels, outputs, average='weighted')
    
    return loss, acc, f1

def inference_fn(model, dataloader, device):
    model.to(device)
    model.eval()
    
    preds = []
    probs = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs_ = model(input_ids, attention_mask=attention_mask).softmax(axis=1)
        
        preds.extend(torch.argmax(outputs_, axis=1).detach().cpu().numpy())
        probs.extend(torch.amax(outputs_, axis=1).detach().cpu().numpy())
        
    return preds, probs