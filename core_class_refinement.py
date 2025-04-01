from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity as cos
import json
import argparse
from collections import defaultdict as ddict
from tqdm import tqdm
import numpy as np
import torch
import os

def main(args):
    model_name = 'all-mpnet-base-v2'
    trans_model = SentenceTransformer(model_name, device=f'cuda:{args.gpu}')

    init_core_classes = json.load(open(os.path.join(args.data_dir, args.dataset, 'train/init_core_classes.json')))
    doc_labels = {doc_id: dic["with ancestors"] for doc_id, dic in init_core_classes.items()}

    label_to_doc_mapping = ddict(list)
    # Iterate through the JSON data to construct the reverse mapping
    for doc_id, labels in doc_labels.items():
        for label_id in labels:
            label_to_doc_mapping[label_id].append(doc_id)

    enriched_file = os.path.join(args.data_dir, args.dataset, 'train/enrichment.txt')
    label_kw_dict = {}
    with open(enriched_file) as file:
        for line in file:
            components = line.strip().split(':')
            node = components[0]
            keywords = components[1]
            keyword_list = keywords.split(',')
            label_kw_dict[node] = keyword_list
    
    llm_enriched_file = os.path.join(args.data_dir, args.dataset, 'train/llm_enrichment.txt')
    with open(llm_enriched_file) as file:
        for line in file:
            components = line.strip().split(':')
            node = components[0]
            keywords = components[1]
            keyword_list = keywords.split(',')
            if node in label_kw_dict.keys():
                label_kw_dict[node] += keyword_list
            else:
                label_kw_dict[node] = keyword_list


    id2label = {}
    label_file_path = os.path.join(args.data_dir, args.dataset, 'labels.txt')
    with open(label_file_path) as label_file:
        for line in label_file:
            label_id, label = line.strip().split('\t')
            id2label[label_id] = label




    doc_id2idx = {}
    idx2doc_id = {}
    all_docs = []

    # Read and tokenize documents
    corpus_path = os.path.join(args.data_dir, args.dataset, 'train/corpus.txt')
    with open(corpus_path) as doc_file:
        for line_id, line in tqdm(enumerate(doc_file)):
            doc_id, doc = line.strip().split('\t')
            idx2doc_id[line_id] = doc_id
            doc_id2idx[doc_id] = line_id
            all_docs.append(doc)
    # pre-calc all document embeddings
    with torch.no_grad():
        doc_embedding = trans_model.encode(all_docs, batch_size=128, 
                                            show_progress_bar=True, convert_to_numpy=True)



    label_id_embed_dict = {}
    # Process label-to-document mapping
    for l_id, d_ids in tqdm(label_to_doc_mapping.items(), desc="Processing Documents"):
        label = id2label[l_id]
        if label not in label_kw_dict or len(label_kw_dict[label]) == 0:
            print('No docs for', l_id, label)
            continue
        keyword_list = label_kw_dict[label]

        label_id2embs = []
        for d_id in d_ids:
            doc = all_docs[doc_id2idx[d_id]]
            for kw in keyword_list:
                if (' ' + doc + ' ').find(' ' + kw.replace('_', ' ') + ' ') != -1:
                        label_id2embs.append(doc_embedding[doc_id2idx[d_id]])
                        break
       
        if len(label_id2embs) == 0:
            print('No docs for', l_id, label)
            continue
        label_id_embed_dict[l_id] = np.mean(label_id2embs, axis=0)

    label_id_map = {}
    label_emb = []
    for lid, emb in label_id_embed_dict.items():
        label_id_map[len(label_emb)] = lid
        label_emb.append(emb)

    # run all doc and sim scores
    doc_sim_mat = cos(doc_embedding, label_emb)


    # calculate conf scores
    doc_id2conf_score = {}
    for idx, sims in tqdm(enumerate(doc_sim_mat)):
        doc_id = idx2doc_id[idx]
        scores = []
        prev = None
        for r, i in enumerate(np.argsort(-sims)):
            if prev is None:
                prev = sims[i]
            else:
                scores.append(prev - sims[i])
                prev = sims[i]
                if len(scores) >= 3:
                    break
        i = np.argmax(scores)
        doc_id2conf_score[doc_id] = (scores[i], i)
    conf_med = np.percentile([s[0] for s in doc_id2conf_score.values()],25)

    reselected_core_classes = {}
    for doc_id, rank_score in sorted(doc_id2conf_score.items(), 
                                     key=lambda x: x[1][0], reverse=True):
        if rank_score[0] < conf_med: break
        sims = doc_sim_mat[doc_id2idx[doc_id]]

        reselected_core_classes[doc_id] = []
        for r, i in enumerate(np.argsort(-sims)):
            reselected_core_classes[doc_id].append(label_id_map[i])
            if r == rank_score[1]:
                break
    print(len(reselected_core_classes))
    json.dump(reselected_core_classes, open(os.path.join(args.data_dir, args.dataset, 'train/refined_core_classes.json'), 'w'), indent=1)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    main(args)