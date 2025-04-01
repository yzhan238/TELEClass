import networkx as nx
import argparse 
import tqdm
from openai import OpenAI
import re
import os
import string
from utils import *

openai.api_key = "YOUR OPENAI API KEY" 
    
def parse_key_terms(original_string):
    """
    Parses a string of comma-separated key terms into a list.

    Args:
    terms_str (str): A string of comma-separated key terms related to hair care.

    Returns:
    list: A list containing individual key terms.
    """
    # Strip whitespace and split the string by commas
    terms_str = original_string.split(":")[1].replace("\n","")
    terms_str = re.sub(r'\d+\.', ',', terms_str)
    key_terms_list = [term.strip().lower().translate(str.maketrans('', '', string.punctuation)) for term in terms_str.split(',')]
    key_terms_list = [i.strip() for i in key_terms_list if i !=""]

    return key_terms_list


def generate_enrichment(class_name,siblings,parent_class):
    out = None
    client = OpenAI(api_key=openai.api_key)


    parent_prompt = f" and is the subclass of class '{', '.join(parent_class)}'" if parent_class!=[] else ""
    if args.dataset == 'Amazon-531':
        message = [
                {"role": "user", "content": f"'{class_name}' is a product class in Amazon{parent_prompt}. Please generate 10 additional key terms about the '{class_name}' class that is relevant to '{class_name}' but irrelevant to {siblings}. Please split the additional key terms using commas and output in the form: Key terms:\n(key terms)."}
                ]
    elif args.dataset == 'DBPedia-298':
        message = [
                {"role": "user", "content": f"'{class_name}' is a article category of Wikipedia articles{parent_prompt}. Please generate 10 additional key terms about the '{class_name}' class that is relevant to '{class_name}' but irrelevant to {siblings}. Please split the additional key terms using commas and output in the form: Key terms:\n(key terms)."}
                ]
    else:
        raise ('No such dataset. You can prepare the generation prompt as the same structure.')
    while out == None:
        try:
            response = client.chat.completions.create(
            model = "gpt-3.5-turbo-0125",
            temperature = 0,
            max_tokens = 3000,
            messages=message
            )
            results = response.choices[0].message.content.strip()

            key_term = parse_key_terms(results)
            key_term=[i.replace(' ', '_') for i in key_term]
            out = class_name.replace(' ', '_')+":"+",".join(key_term)
        except:
            print('generation error')
    
    return out

def get_sib(root:Node,idx):
    cur_node = root.findChild(idx)
    par_node = cur_node.parents
    sib = []
    for j in par_node:
        sib.extend([i for i in j.childs if i != cur_node])
    return [i.name.replace("_"," ") for i in sib]

def get_par(root:Node,idx):
    cur_node = root.findChild(idx)
    parent_list = []
    # assert len(cur_node.parents)==1
    for par_node in cur_node.parents:
        if par_node != root:
            parent_list.append(par_node.name)
            for par_par_node in par_node.parents:
                if par_par_node != root:
                    parent_list.append(par_par_node.name)
    return [i.replace("_"," ") for i in parent_list]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    args = parser.parse_args()

    # read taxonomy graph
    root, id2label, label2id = createGraph(os.path.join(args.data_dir, args.dataset))
    num_class = len(id2label)

    # generate LLM-based taxonomy enrichment
    enrichment_list = []
    for i in tqdm.tqdm(range(num_class)):
        siblings = get_sib(root,str(i))
        class_name = root.findChild(str(i)).name.replace("_"," ")
        parent_class = get_par(root,str(i))
        enrichment = generate_enrichment(class_name,siblings,parent_class)
        enrichment_list.append(enrichment)

    OUTPUT = f'{args.data_dir}/{args.dataset}/train/llm_enrichment.txt'
    with open(OUTPUT, 'w') as fout1:
        for l in enrichment_list:
            fout1.write(l+'\n')

    