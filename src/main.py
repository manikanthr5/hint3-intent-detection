import os
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

from transformers import AdamW
from transformers import DistilBertTokenizer

from data import get_train_data, get_test_data, HINTDataset
from models import HINTModel
from utils import train_fn, val_fn, inference_fn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

def parse_arguments(args):
    parser = ArgumentParser()
    parser.add_argument("random_seed", '-rs', required=False, default=42, type=int)
    parser.add_argument("epochs", '-ep', required=False, default=20, type=int)
    parser.add_argument("learning_rate", '-lr', required=False, default=5e-5, type=float)
    parser.add_argument("max_steps", '-ms', required=False, default=3, type=int)
    parser.add_argument("model_name", '-mn', required=False, default="distill_bert.pth")
    parser.add_argument("model_dir", '-md', required=False, default='../models/')
    parser.add_argument("test_file_name", '-tfn', required=False, default="distill_bert_test_results.csv")
    parser.add_argument("output_dir", '-od', required=False, default="../output/")
    parser.add_argument("min_prob", '-mp', required=False, default=0.25, type=float)
    return vars(parser.parse_args(args))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    args = parse_arguments(sys.argv[1:])
    set_seed(args['random_seed'])
    df = get_train_data()
    test_df = get_test_data()
    NUM_CLASSES = df['label'].nunique()

    train_texts, val_texts, train_labels, val_labels = train_test_split(df['sentence'], df['label_int'], random_state=args['random_seed'], test_size=.2)
    print(train_texts.shape, val_texts.shape, train_labels.shape, val_labels.shape)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts.to_list(), truncation=True, padding=True)
    val_encodings = tokenizer(val_texts.to_list(), truncation=True, padding=True)
    test_encodings = tokenizer(test_df['sentence'].to_list(), truncation=True, padding=True)

    train_dataset = HINTDataset(train_encodings, train_labels.values)
    val_dataset = HINTDataset(val_encodings, val_labels.values)
    test_dataset = HINTDataset(test_encodings, test_df['label_int'].values)

    model = HINTModel(num_classes=NUM_CLASSES)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.ffn.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    optim = AdamW(model.parameters(), lr=args['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    best_acc = 0

    Path(args['model_dir']).mkdir(parents=True, exist_ok=True)

    for epoch in range(args['epochs']):
        train_loss, train_acc, train_f1 = train_fn(model, train_loader, loss_fn, optim, device)
        val_loss, val_acc, val_f1 = val_fn(model, val_loader, loss_fn, device)
        print(f"{epoch+1}: train: [{train_loss:.3f}, {train_acc:.3f}, {train_f1:.3f}], val: [{val_loss:.3f}, {val_acc:.3f}, {val_f1:.3f}]")
        if val_acc > best_acc:
            best_acc = val_acc
            step = 0
            torch.save(model.state_dict(), f"{args['model_dir']}/{args['model_path']}")
        else:
            step += 1
        if step >= args['max_steps']:
            break

    model.load_state_dict(torch.load(f"{args['model_dir']}/{args['model_path']}", map_location=device))
    print("model successfully loaded!")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    preds, probs = inference_fn(model, test_loader, device)
    test_df['preds'] = preds
    test_df['probs'] = probs
    test_df['label_int'] = test_df['label_int'].fillna(NUM_CLASSES + 1)
    test_df['updated_preds'] = test_df['preds']
    test_df.loc[test_df['probs'] <= args['min_prob'], 'updated_preds'] = NUM_CLASSES + 1

    Path(args['output_dir']).mkdir(parents=True, exist_ok=True)
    test_df.to_csv(f"{args['output_dir']}/{args['test_file_name']}", index=False)
    
    acc1 = accuracy_score(test_df['label_int'], test_df['preds'])
    acc2 = accuracy_score(test_df['label_int'], test_df['updated_preds'])

    f11 = f1_score(test_df['label_int'], test_df['preds'], average='weighted')
    f12 = f1_score(test_df['label_int'], test_df['updated_preds'], average='weighted')

    print(f"Default: acc: {acc1}, f1_score: {f11}")
    print(f"Updated with Min Prob: acc: {acc2}, f1_score: {f12}")

if __name__ == "__main__":
    main()