import networkx as nx
import argparse 
import tqdm
import  time
import json
import openai
from openai import OpenAI
import requests

openai.api_key = "YOUR OPENAI API KEY" 

def generate_pseudo_docs(path):
    backslash_char = "\\"
    client = OpenAI(api_key=openai.api_key)
    out = []

    if args.dataset == 'Amazon-531':
        message = [
                {"role": "user", "content": f"""\
                Suppose you are an Amazon Reviewer, please generate 5 various and reliable passages following the requirements below:
                1. Must generate reviews following the themes of the taxonomy path: {path} but without mentioning {path}
                2. Must be in length above 100 words
                3. The writing style and format of the text should be a product review
                4. Should keep the generated text to be diverse, specific, and consistent with the given taxonomy path.
                You should focus on {path.split(backslash_char)[-1]} as the documents"""},
                {"role": "user", "content": "Your results should in be a json string in the format like \"[{\"doc\": text}]\" text is each generated document."}
                ]
    elif args.dataset == 'DBPedia-298':
        message = [
                {"role": "user", "content": f"""\
                Suppose you are a Wikipedia Contributor, please generate 5 various and reliable passages following the requirements below:
                1. Must generate articles following the themes of the taxonomy path: {path} but without mentioning {path}
                2. Must be in length above 300 words
                3. The writing style and format of the text should be a Wikipedia page.
                4. Should keep the generated text to be diverse, specific, and consistent with the given taxonomy path.
                You should focus on {path.split(backslash_char)[-1]} as the documents"""},
                {"role": "user", "content": "Your results should in be a json string in the format like \"[{\"doc\": text}]\" text is each generated document."}
                ]
    else:
        raise ('No such dataset. You can prepare the generation prompt as the same structure.')
    while len(out) == 0:
        try:
            response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            temperature = 1,
            max_tokens = 3000,
            messages=message
            )
            results = response.choices[0].message.content.strip()
            results_json = json.loads(results)
            for review in results_json:
                 out.append(review['doc'])
        except:
            print('generation error')
    
    return out

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

def get_all_paths(graph, start_node):
    leaf_nodes = [node for node in graph.nodes() if graph.out_degree(node) == 0]

    all_paths = []
    for leaf_node in leaf_nodes:
        paths = list(nx.all_simple_paths(graph, source=start_node, target=leaf_node))
        for path in paths:
            path.pop(path.index('root'))
            all_paths.append('\\'.join(path))
    return all_paths

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    args = parser.parse_args()

    # read labels
    with open(f'{args.data_dir}/{args.dataset}/labels.txt', 'r') as file:
        id2label = {}
        label2id = {}
        each_lines = file.read().split('\n')
        for each_line in each_lines:
            if each_line == '':
                continue
            id2label[each_line.split('\t')[0]] = each_line.split('\t')[1]
            label2id[each_line.split('\t')[1]] = each_line.split('\t')[0]

    # read the taxonomy started from the root
    taxo = read_taxonomy(f'{args.data_dir}/{args.dataset}/label_hierarchy.txt')
    paths = get_all_paths(taxo, 'root')

    id_docs = {}
    doc_id_labels = {}

    num_doc = 0
    with open(f'{args.data_dir}/{args.dataset}/train/generated_docs.txt', 'w') as doc_file:
        for path in tqdm.tqdm(paths):
            path_list = path.split('\\')
            doc_pool = []
            while len(doc_pool) < 5:
                try:
                    results = generate_pseudo_docs(path)
                    for doc in results:
                        if len(doc_pool) == 5:
                            break
                        if len(doc) != 0:
                            # avoids some common generation errors
                            if ":" in doc or "=" in doc or len(doc) < 100:
                                continue
                            doc_pool.append(doc)
                except openai.APITimeoutError:
                    print('Timeout!')
                    continue
                except openai.RateLimitError:
                    print('RateLimitError')
                    time.sleep(10)
                    continue
            time.sleep(0.1)

            for doc in doc_pool:
                id_docs[str(num_doc)] = doc
                local_doc_to_label = {'core_classes': label2id[path_list[-1]], 'with ancestors': \
                                                            [label2id[label] for label in path_list]}
                doc_id_labels[str(num_doc)] = local_doc_to_label
                
                doc_file.write(str(num_doc) + '\t' + id_docs[str(num_doc)].replace('\n', ' ') + '\n')
                with open(f'{args.data_dir}/{args.dataset}/train/generated_doc2label.json', 'w') as label_file:
                    json.dump(doc_id_labels, label_file)
                    num_doc += 1
