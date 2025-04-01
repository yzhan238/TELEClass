from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os
import copy
import sys
from tqdm import tqdm
import argparse
from openai import OpenAI
import openai
import time
import queue
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos
import math
from utils import *


def tree_search_bfs(root, document_text, gpu, doc_emb, key_term_emb_dict):
    queue = [root]  # current level queue
    next_queue = [] # next level queue  
    similarity_score_dic = {}
    level = 0       # current level
    while len(queue) > 0:
        ## for each class at level l in the queue
        for node in queue:
            ## calculate <document, class> similarity score for all its child
            childs = node.childs
            if len(childs) == 0:
                continue
            c2s = {}
            for cur_child in childs:
                cur_name = cur_child.name
                cur_child_embed = key_term_emb_dict[cur_name]
                scores = max(cos(doc_emb.reshape(1,-1), cur_child_embed).tolist()[0])
                c2s[cur_name]= math.exp(scores)

            for child in childs:
                child.similarity_score = c2s[child.name]
                cur_parents = child.parents
                cur_parents.sort(key=lambda x: x.path_score, reverse=True)
                child.path_score = child.similarity_score * cur_parents[0].path_score

            ## select l + 3 classes from its children classes that are most similar to D
            childs.sort(key=lambda x: x.similarity_score, reverse=True)
            next_queue.extend(childs[:min(len(childs), level + 3)])

        ## after processing all node at current level, proceed to next level
        if level == 0:
            queue = next_queue
        else:
            next_queue.sort(key=lambda x: x.path_score, reverse=True)
            queue = next_queue[:(level+2)**2]

        for child in queue:
            similarity_score_dic[child.node_id] = child.similarity_score

        next_queue = []
        level += 1
    # return a list of class names that are selected
    return similarity_score_dic

def process_document(root_node, id2label, label2id, document_text, document_id, client, gpt_template, gpu, doc_emb, key_term_emb_dict):
    # make a tree copy
    root = copy.deepcopy(root_node)
    # set root path score to 1
    root.path_score = 1
    root.similarity_score = 1
    # process all level, calculate path score
    sim_dic = tree_search_bfs(root, document_text, gpu, doc_emb, key_term_emb_dict)


    class_names = [id2label[cid].replace('_', ' ') for cid in sim_dic]
    instruction = gpt_template.format(', '.join(class_names))
    response = api_call(client, document_text, instruction, demos=[], temperature=0)
    class_names = [cn.replace(' ', '_') for cn in response.split(', ')]
    classes = [label2id[cn] for cn in class_names if cn in label2id]

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

    return {'response':response, 
            'core classes':classes, 
            'with ancestors': list(class_set)}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()


    os.environ["OPENAI_API_KEY"] = 'YOUR OPENAI API KEY'
    client = OpenAI()

    enriched_file = os.path.join(args.data_dir, args.dataset, 'train/llm_enrichment.txt')
    label_keyterm_dict = {}
    with open(enriched_file) as file:
        for line in file:
            components = line.strip().split(':')
            node = components[0]
            keywords = components[1]
            keyword_list = keywords.split(',')
            label_keyterm_dict[node] = keyword_list

    if args.dataset == 'Amazon-531':
        gpt_template = 'You will be provided with an Amazon product review, and please select'\
                       ' its product types from the following categories: {}.'\
                       ' Just give the category names as shown in the provided list.'
    elif args.dataset == 'DBPedia-298':
        gpt_template = 'You will be provided with a Wikipedia article describing an entity at the beginning,'\
                       ' and please select the entity types from the following categories: {}. Just give the category names as shown in the provided list.'
    else:
        print('Unknown dataset! Please specify its prompt templates.')


    model_name = 'all-mpnet-base-v2'
    model = SentenceTransformer(model_name, device=f'cuda:{args.gpu}') 
    # construct graph from file
    root, id2label, label2id = createGraph(os.path.join(args.data_dir, args.dataset))
    corpus_path = os.path.join(args.data_dir, args.dataset, 'train/corpus.txt')
    num_line = get_num_lines(corpus_path)
    num_class = len(id2label)
    writing_result = {}


    all_docs = [] 
    all_docs_id = []
    with open(corpus_path) as f:
        for i, line in tqdm(enumerate(f), total=num_line):
            ## get current line and process document
            doc_id, doc = line.strip().split('\t')
            all_docs.append(doc)
            all_docs_id.append(doc_id)
    with torch.no_grad():
        total_doc_embedding = model.encode(all_docs, batch_size=128, 
                                        show_progress_bar=True, convert_to_numpy=True)
        
    key_term_emb_dict = {}
    for i in tqdm(range(num_class)):
        current_label = id2label[str(i)]
        current_key = [current_label]+label_keyterm_dict[current_label]
        current_key = [i.replace('_', ' ') for i in current_key]
        current_embed = model.encode(current_key, batch_size=128, convert_to_numpy=True)
        key_term_emb_dict[current_label] = current_embed

    
    
    
    for index,(doc_id,doc) in tqdm(enumerate(zip(all_docs_id,all_docs))):
        ## process document
        doc_emb=total_doc_embedding[index]
        writing_result[doc_id] = process_document(root, id2label, label2id, doc, doc_id, client, gpt_template, args.gpu, doc_emb, key_term_emb_dict)
    json.dump(writing_result, open(os.path.join(args.data_dir, args.dataset, 'train/init_core_classes.json'), 'w'), indent=1)
