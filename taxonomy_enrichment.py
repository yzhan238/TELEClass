from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import networkx as nx  
from tqdm import tqdm
import numpy as np
import argparse
import json
import math
import torch
import pickle
import os

#read taxonomy file
def read_taxonomy(taxonomy_file):
    # Create a Directed Graph
    graph = nx.DiGraph()
    # Read the file and add nodes and edges to the graph
    with open(taxonomy_file, "r") as file:
        for line in file:
            if line == '\n':
                continue
            components = line.strip().split('	')
            # Iterate through the components to add nodes and edges
            source_node = id2label[components[0]]
            target_node = id2label[components[1]]

            # Add nodes and edges
            graph.add_node(source_node)
            graph.add_node(target_node)
            graph.add_edge(source_node, target_node)
        no_parents = [node for node in graph.nodes if graph.in_degree(node) == 0]
        if len(no_parents) != 0:
            graph.add_node('root')
            for np in no_parents:
                graph.add_edge('root', np)
    return graph

# Function to get sibling nodes
def get_siblings(graph, node):
    parents = list(graph.predecessors(node))  # Get the parent of the node
    siblings = []
    for parent in parents:
        sibling = set()
        for child in graph.successors(parent):
            sibling.add(child)
        siblings.append(sibling)
    return list(siblings)

# Function to get parent nodes
def get_parents(graph, node):
    return list(graph.predecessors(node))

def cosine_similarity(a, b):
    # Calculate the dot product of A and B
    dot_product = np.dot(a, b)

    # Calculate the Euclidean norms of A and B
    norm_A = np.linalg.norm(a)
    norm_B = np.linalg.norm(b)

    # Calculate the cosine similarity
    cosine_similarity = dot_product / (norm_A * norm_B)
    if cosine_similarity < 0:
        cosine_similarity = 0
    return  cosine_similarity

def BM25(df, maxdf, tf, dl, avgdl, k=1.2, b=0.5):
    score = tf * (k + 1) / (tf + k * (1 - b + b * (dl / avgdl)))
    df_factor = math.log(1 + df, 2) / math.log(1 + maxdf, 2)
    score *= df_factor
    return score


def Softmax(score_list):
    exp_sum = 1
    for score in score_list:
        exp_sum += math.exp(score)
    exp_list = [math.exp(x) / exp_sum for x in score_list]
    return exp_list

 
def document_retrieval(doc_path, node_id_path, seeds):
    with open(doc_path) as file:
        docs = file.readlines()
    with open(node_id_path) as class_file:
        doc_labels = json.load(class_file)
    
    confident_docs = []

    for node in seeds:
        local_docs = {}
        local_docs['seed'] = node
        local_docs['sentences'] = []
        for doc_id in doc_labels:
                if label2id[node] in doc_labels[doc_id]['with ancestors']:
                    local_docs['sentences'] .append(docs[int(doc_id)])
        
        confident_docs.append(local_docs)
    return confident_docs

def bert_embed(text, model, tokenizer, device):
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(text, max_length=256, truncation=True)).unsqueeze(0).to(device)
        outputs = model(input_ids)
        hidden_states = outputs[2][-1][0]
        emb = torch.mean(hidden_states, dim=0).cpu()
    return emb

#create word embeddings for future usage
def create_vocab_embeddinngs(seeds, doc_path):
    word_embed_dict = {}
    print('Start to Create Word Embeddings')
    with open(doc_path, 'r') as file:
        words = set(word_tokenize(file.read()))
        for word in tqdm(words):
            word_embed_dict[word] = bert_embed(word.replace('_', ' '),model, tokenizer, device)
        for seed in seeds:
            word_embed_dict[seed] = bert_embed(seed.replace('_', ' '), model, tokenizer, device)
    return word_embed_dict

#extract all sets of siblings from a taxonomy
def extract_sibling_comb(taxonomy):
    sibling_sets = []
    
    for node in list(nx.topological_sort(taxo)):
        sibling_tmp_set = get_siblings(taxo, node)
        for sibling_set in sibling_tmp_set:
            if sibling_set not in sibling_sets:
                sibling_sets.append(sibling_set)
    
    return sibling_sets

def local_caseolap(top_sentences, seeds, word_emb_dict, topk=20):
    # experiment on named entity recognition
    n = len(seeds)
    tf = [defaultdict(int) for _ in range(n)]
    df = [defaultdict(int) for _ in range(n)]
    for idx, data in enumerate(top_sentences):
        for sent in data['sentences']:
            words = sent.split()
            for word in words:
                tf[idx][word] += 1
            words = set(words)
            for word in words:
                df[idx][word] += 1

    stop_words = set(stopwords.words('english'))
    candidate = set()
    for idx in range(n):
        for word in tf[idx]:
            if tf[idx][word] >= 5 and word not in stop_words:
                candidate.add(word)
    maxdf = [max(df[x].values()) for x in range(n)]
    dl = [sum(tf[x].values()) for x in range(n)]
    avgdl = sum(dl) / len(dl)
    bm25 = [defaultdict(float) for _ in range(n)]
    for idx in range(n):
        for word in candidate:
            bm25[idx][word] = BM25(df[idx][word], maxdf[idx], tf[idx][word], dl[idx], avgdl)

    dist = {}
    for word in candidate:
        dist[word] = Softmax([bm25[x][word] for x in range(n)])
    
    # get quality phrases pool
    phrase_scores = {}
    with open(PHRASES, 'r') as phrases_file:
        phrases_raw = phrases_file.readlines()
    for phrase_score in phrases_raw:
        phrase_scores[phrase_score.split('\t')[1][:-1].replace(' ', '_')] = float(phrase_score.split('\t')[0])


    #top terms for each node on taxonomy
    node_top_terms = {}
    caseolap_scores = {}
    for idx in range(n):
        seed = seeds[idx]
        caseolap = {}

        #bert_sim
        seed_embed = word_emb_dict[seed]
        for word in candidate:
            if word not in phrase_scores:
                continue
            pop = math.log(1 + df[idx][word], 2)
            if word in word_emb_dict:
                word_embed = word_emb_dict[word]
            else:
                word_embed = bert_embed(word.replace('_', ' '), model, tokenizer, device)
                word_emb_dict[word] = word_embed
            
            caseolap[word] = (pop ** 0.2 * (dist[word][idx])**0.8) * (float(cosine_similarity(seed_embed, word_embed)))
        
        caseolap_sorted = sorted(caseolap.items(), key=lambda x: x[1], reverse=True)
        top_terms = caseolap_sorted[:topk]
        node_top_terms[seed] = set([w[0] for w in top_terms])
        # get caseolap scores to select key phrases
        caseolap_scores[seed] = caseolap
    
    # Discriminative keywords on each sibling set
    for seed in seeds:
        for sib in seeds:
            if sib != seed:
                shared_kws = node_top_terms[seed].intersection(node_top_terms[sib])
                for kw in shared_kws:
                    seed_score = caseolap_scores[seed][kw]
                    sib_score = caseolap_scores[sib][kw] if kw in caseolap_scores[sib] else 0
                    if seed_score < sib_score:
                        node_top_terms[seed].discard(kw)
                    else:
                        node_top_terms[sib].discard(kw)
    
    return node_top_terms

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    CORPUS = f'{args.data_dir}/{args.dataset}/train/phrasal_corpus.txt'
    OUTPUT = f'{args.data_dir}/{args.dataset}/train/enrichment.txt'
    TAXO = f'{args.data_dir}/{args.dataset}/label_hierarchy.txt'
    PHRASES = f'{args.data_dir}/{args.dataset}/train/autophrase_results.txt'
    CLASSIFICATION = f'{args.data_dir}/{args.dataset}/train/init_core_classes.json'
    
    bert_model = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    device = f'cuda:{args.gpu}'
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model = BertModel.from_pretrained(bert_model, output_hidden_states=True).to(device)

    # read class labels
    with open(f'{args.data_dir}/{args.dataset}/labels.txt', 'r') as file:
        id2label = {}
        label2id = {}
        each_lines = file.read().split('\n')
        for each_line in each_lines:
            if each_line == '':
                continue
            id2label[each_line.split('\t')[0]] = each_line.split('\t')[1]
            label2id[each_line.split('\t')[1]] = each_line.split('\t')[0]
    
    # read the taxonomy for the enrich
    taxo = read_taxonomy(TAXO)
    nodes = taxo.nodes()

    # construct a quality phrase pool
    phrases_pool = []
    with open(PHRASES, 'r') as phrases_file:
        phrases_raw = phrases_file.readlines()
        for phrase_score in phrases_raw:
            phrases_pool.append(phrase_score.split('\t')[1][:-1].replace(' ', '_'))

    # load/create word embedding
    embed_path = f'{args.data_dir}/{args.dataset}/train/word_embed_avg.pkl'
    if not os.path.exists (embed_path):
        word_embeds = create_vocab_embeddinngs(taxo.nodes(), CORPUS)
        with open(embed_path, 'wb') as file:
            pickle.dump(word_embeds, file)
    else:
        with open(embed_path, 'rb') as file:
            word_embeds = pickle.load(file)
    

    global_seed_topics = {}
    for siblings in tqdm(extract_sibling_comb(taxo)):
        contrast_nodes = set(siblings)
        for node in siblings:
            if node not in nodes:
                contrast_nodes.remove(node)
        if len(contrast_nodes) == 0:
            continue
        contrast_nodes = list(contrast_nodes)
        top_sentences = document_retrieval(CORPUS, CLASSIFICATION, contrast_nodes)
        top_sentences_filtered = []
        contrast_filtered_nodes = []
        for data in top_sentences:
            if len(data['sentences']) > 0:
                contrast_filtered_nodes.append(data['seed'])
                top_sentences_filtered.append(data)
        if len(contrast_filtered_nodes) == 0:
            continue
        local = local_caseolap(top_sentences_filtered, contrast_filtered_nodes, word_embeds, topk=20)
        for seed in local:
            if seed not in global_seed_topics:
                global_seed_topics[seed] = local[seed]
            else:
                global_seed_topics[seed] = global_seed_topics[seed].intersection(local[seed])
    
    with open(OUTPUT, 'w') as fout1:
        for seed in global_seed_topics:
            top_terms = global_seed_topics[seed]
            fout1.write(seed+':'+','.join(top_terms)+'\n')
