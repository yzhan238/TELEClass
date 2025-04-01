from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import *
import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()

    data_dict = torch.load(os.path.join(args.data_dir, args.dataset, 'train/training_data.pt'))
    dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"], data_dict['sample_mask'])
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    class_emb = torch.load(os.path.join(args.data_dir, args.dataset, 'class_emb.pt'))

    model = ClassModel('bert-base-uncased', 768, class_emb).to(f'cuda:{args.gpu}')

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 5e-6,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    loss_fn = multilabel_bce_loss_w


    model.zero_grad()
    for e in range(args.epoch):
        print(f'Training epoch: {e}')
        total_train_loss = 0
        for j, batch in enumerate(tqdm(data_loader)):
            input_ids = batch[0].to(f'cuda:{args.gpu}')
            input_mask = batch[1].to(f'cuda:{args.gpu}')
            labels = batch[2].to(f'cuda:{args.gpu}')
            sample_mask = batch[3].to(f'cuda:{args.gpu}')
            output = model(input_ids, 
                           input_mask)
            
            loss = loss_fn(output, labels, sample_mask)
            total_train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()

        print(f'Epoch {e}', total_train_loss / j)

    torch.save(model.state_dict(), os.path.join(args.data_dir, args.dataset, f'train/model.pt'))
    