import argparse
from transformers import AutoTokenizer
from prepare_training_data import create_dataset
from model import ClassModel
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import os
import itertools
from tqdm import tqdm

def calculate_ranks_from_similarities(all_similarities, positive_relations):
    """
    all_similarities: a np array
    positive_relations: a list of array indices

    return a list
    """
    positive_relation_similarities = all_similarities[positive_relations]
    negative_relation_similarities = np.ma.array(all_similarities, mask=False)
    negative_relation_similarities.mask[positive_relations] = True
    ranks = list((negative_relation_similarities >= positive_relation_similarities[:, np.newaxis]).sum(axis=1) + 1)
    return ranks


def precision_at_k(preds, gts, k=1):
    assert len(preds) == len(gts), "number of samples mismatch"
    p_k = 0.0
    for pred, gt in zip(preds, gts):
        p_k += ( len(set(pred[:k]) & set(gt)) / k ) 
    p_k /= len(preds)
    return p_k


def mrr(all_ranks):
    """ Scaled MRR score, check eq. (2) in the PinSAGE paper: https://arxiv.org/pdf/1806.01973.pdf
    """
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    scaled_rank_positions = np.ceil(rank_positions)
    return (1.0 / scaled_rank_positions).mean()


def example_f1(trues, preds):
    """
    trues: a list of true classes
    preds: a list of model predicted classes
    """
    f1_list = []
    for t, p in zip(trues, preds):
        f1 = 2 * len(set(t) & set(p)) / (len(t) + len(p))
        f1_list.append(f1)
    return np.array(f1_list).mean()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--model_pth', type=str, help='model ckpt')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    class_emb = torch.load(os.path.join(args.data_dir, args.dataset, 'class_emb.pt'))
    model = ClassModel('bert-base-uncased', 768, class_emb).to(f'cuda:{args.gpu}')
    model.load_state_dict(torch.load(args.model_pth, map_location=f'cuda:{args.gpu}'))


    corpus = {}
    with open(os.path.join(args.data_dir, args.dataset, 'test/corpus.txt')) as f:
        for line in f:
            i, t = line.strip().split('\t')
            corpus[i] = t

    test_data, id_list = create_dataset(corpus, tokenizer)
    dataset = TensorDataset(test_data["input_ids"], test_data["attention_masks"])
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        predictions = []
        for batch in tqdm(data_loader):
            input_ids = batch[0].to(f'cuda:{args.gpu}')
            input_mask = batch[1].to(f'cuda:{args.gpu}')
            output = model(input_ids, input_mask).cpu().numpy()
            predictions.append(output)
    predictions = np.concatenate(predictions, axis=0)

    gt_labels = {}
    with open(os.path.join(args.data_dir, args.dataset, 'test/doc2labels.txt')) as f:
        for line in f:
            i, t = line.strip().split('\t')
            gt_labels[i] = t.split(',')
    gt_labels = [list(map(int, gt_labels[i])) for i in id_list]

    all_ranks = []
    top_classes = []
    for pred, gt in zip(predictions, gt_labels):
        all_ranks.append(calculate_ranks_from_similarities(pred, gt))
        top_classes.append(np.argsort(-pred)[:3])

    for k in [1, 2, 3]:
        print(f"Precision@{k}: {precision_at_k(top_classes, gt_labels, k)}")
    print(f"MRR: {mrr(all_ranks)}")

    print(f"Exmaple F1: {example_f1(gt_labels, top_classes)}")