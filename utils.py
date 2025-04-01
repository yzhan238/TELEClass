import os
import mmap
import openai


def api_call(client, doc, instruction, demos=[], temperature=None, model='gpt-3.5-turbo-0125'):
    '''
    doc str: query
    instruction str: system instruction
    demos List((str, str)): demonstrations, if any
    temperature: None for default temprature
    '''
    
    messages = [{"role": "system", "content": instruction}]
    
    for demo_doc, demo_label in demos[::-1]:
        messages.append({"role": "user", "content": demo_doc})
        messages.append({"role": "assistant", "content": demo_label})
    
    messages.append({"role": "user", "content": doc})
    
    timeout_num = 0
    while timeout_num < 3:
        try:
            if temperature is not None:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            break
        except openai.APITimeoutError:
            print('Timeout!')
            timeout_num += 1
            continue
        except openai.RateLimitError:
            print('RateLimitError')
            time.sleep(10)
            continue
    time.sleep(0.1)
    if timeout_num >= 3:
        return ''
    
    return completion.choices[0].message.content


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


class Node:
    def __init__(self, node_id, name):
        self.name = name
        self.node_id = node_id
        self.parents = []
        self.childs = []
        self.similarity_score = 0
        self.path_score = 0

    def addChild(self, child):
        if child not in self.childs:
            self.childs.append(child)

    def addParent(self, parent):
        if parent not in self.parents:
            self.parents.append(parent)

    def findChild(self, node_id):
        if node_id == self.node_id:
            return self
        if len(self.childs) == 0:
            return None
        for child in self.childs:
            ans = child.findChild(node_id)
            if ans != None:
                return ans
        return None
    

#create taxonomy graph
def createGraph(file_addr):
    root = None
    # sanity check if file exist
    if not os.path.exists(file_addr):
        print(f"ERROR. Taxonomy file addr {file_addr} not exists.")
        exit(-1)

    id2label = {}
    label2id = {}
    with open(os.path.join(file_addr, 'labels.txt')) as f:
        for line in f:
            label_id, label_name = line.strip().split('\t')
            id2label[label_id] = label_name
            label2id[label_name] = label_id

    # construct graph from file
    with open(os.path.join(file_addr, 'label_hierarchy.txt')) as f:
        root = Node(-1, 'ROOT')
        for line in f:
            parent_id, child_id = line.strip().split('\t')
            parent = id2label[parent_id]
            child = id2label[child_id]
            parent_node = root.findChild(parent_id)
            if parent_node is None:
                parent_node = Node(parent_id, parent)
                root.addChild(parent_node)
                parent_node.addParent(root)
            child_node = root.findChild(child_id)
            if child_node is None:
                child_node = Node(child_id, child)
            parent_node.addChild(child_node)
            child_node.addParent(parent_node)
    
    return root, id2label, label2id