from joblib import Parallel, delayed
import torch
import os
from math import ceil
from tqdm import tqdm
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import f1_score
from utils import Node, createGraph
import queue
from transformers import AutoTokenizer, AutoModel
import argparse
import json


def encode(docs, tokenizer, max_len):
    encoded_dict = tokenizer.batch_encode_plus(docs, add_special_tokens=True, 
                                               max_length=max_len, padding='max_length',
                                                return_attention_mask=True, truncation=True, 
                                               return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

def create_dataset(id2doc, tokenizer, loader_file=None, doc2parents=None, doc2child=None, max_len=512, num_cpus=20):
    docs = []
    labels = []
    sample_mask = []
    id_list = []
    if doc2parents is not None and doc2child is not None:
        for doc_id in doc2parents:
            docs.append(id2doc[doc_id])
            labels.append(doc2parents[doc_id])
            sample_mask.append(doc2child[doc_id])
            id_list.append(doc_id)
    else:
        for doc_id in id2doc:
            docs.append(id2doc[doc_id])
            id_list.append(doc_id)
    print(f"Converting texts into tensors.")
    chunk_size = ceil(len(docs) / num_cpus)
    chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
    results = Parallel(n_jobs=num_cpus)(delayed(encode)(docs=chunk, tokenizer=tokenizer, max_len=max_len) for chunk in chunks)
    input_ids = torch.cat([result[0] for result in results])
    attention_masks = torch.cat([result[1] for result in results])
    if doc2parents is not None and doc2child is not None:
        labels = torch.tensor(labels)
        sample_mask = torch.tensor(sample_mask)
        data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels, "sample_mask":sample_mask}
    else:
        data = {"input_ids": input_ids, "attention_masks": attention_masks}
    if loader_file is not None:
        print(f"Saving encoded texts into {loader_file}")
        torch.save(data, loader_file)
    return data, id_list

def construct_samples(doc2classes, root, id2label):

    doc2parents = {}
    for doc_id, classes in doc2classes.items():
        q = queue.Queue()
        class_set = set(classes)
        for c in classes:
            q.put(root.findChild(c))
        while not q.empty():
            c = q.get()
            for p in c.parents:
                pid = p.node_id
                if pid == -1: continue
                q.put(p)
                class_set.add(pid)
        doc2parents[doc_id] = [(1 if str(c) in class_set else 0) for c in range(len(id2label))]


    doc2child = {}
    for doc_id, classes in doc2classes.items():
        q = queue.Queue()
        class_set = set()
        for c in classes:
            q.put(root.findChild(c))
        while not q.empty():
            c = q.get()
            for p in c.childs:
                pid = p.node_id
                if pid in classes: continue
                q.put(p)
                class_set.add(pid)
        doc2child[doc_id] = [(0 if str(c) in class_set else 1) for c in range(len(id2label))]

    return doc2parents, doc2child


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # construct graph from file
    root, id2label, label2id = createGraph(os.path.join(args.data_dir, args.dataset))

    if not os.path.exists(os.path.join(args.data_dir, args.dataset, 'class_emb.pt')):
        encoder = AutoModel.from_pretrained('bert-base-uncased').to(f'cuda:{args.gpu}')
        class_emb = []
        for i in range(len(id2label)):
            l = id2label[str(i)]
            inputs = tokenizer(l.replace('_', ' '), return_tensors="pt")
            with torch.no_grad():
                output = encoder(torch.tensor(inputs['input_ids'], device=f'cuda:{args.gpu}')).last_hidden_state
            emb = output[0, 1:-1].mean(dim=0).cpu()
            class_emb.append(emb)
        class_emb = torch.stack(class_emb)
        print(class_emb.size())
        torch.save(class_emb, os.path.join(args.data_dir, args.dataset, 'class_emb.pt'))

    doc2classes = json.load(open(os.path.join(args.data_dir, args.dataset, 'train/refined_core_classes.json')))

    doc2parents, doc2child = construct_samples(doc2classes, root, id2label)

    generated_doc2label = json.load(open(os.path.join(args.data_dir, args.dataset, 'train/generated_doc2label.json')))
    scaling = float(len(doc2classes)) / len(generated_doc2label)
    for doc_id, class_dict in generated_doc2label.items():
        g_doc_id = 'g'+doc_id
        doc2parents[g_doc_id] = [(1 if str(c) in set(class_dict['with ancestors']) else 0) for c in range(len(id2label))]
        doc2child[g_doc_id] = [scaling]*len(id2label)

    corpus = {}
    with open(os.path.join(args.data_dir, args.dataset, 'train/corpus.txt')) as f:
        for line in f:
            i, t = line.strip().split('\t')
            corpus[i] = t
    with open(os.path.join(args.data_dir, args.dataset, 'train/generated_docs.txt')) as f:
        for line in f:
            doc_id, doc = line.strip().split('\t')
            corpus['g'+doc_id] = doc


    create_dataset(corpus, tokenizer, os.path.join(args.data_dir, args.dataset, 'train/training_data.pt'), doc2parents, doc2child, max_len=512, num_cpus=20)

