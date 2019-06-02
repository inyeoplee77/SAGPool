from glob import glob
import tensorflow as tf
import argparse
import os
from tqdm import tqdm
from ast import literal_eval

import numpy as np

def get_data_dictionary(files):
    datasets = dict()
    for f in files:
        file_name = f.split(os.sep)[-1]
        if file_name == 'README.txt':
            continue
        dataset = file_name.split('_')[0]
        if dataset not in datasets:
            datasets[dataset] = dict()
        key = file_name.replace(dataset+"_",'').replace('.txt','')
        datasets[dataset][key] = f
    return datasets

def process_line(line):
    line = line.rstrip().split(',')
    if len(line) == 1:
        return literal_eval(line[0].strip())
    # if type(literal_eval(line[0].strip())) == float:
    #     print(line[0].strip(),literal_eval(line[0].strip()),type(literal_eval(line[0].strip())))
    return [literal_eval(x.strip()) for x in line]


def process_data(dataset,name=""):
    
    A = [process_line(x) for x in tqdm(open(dataset['A']).readlines(),desc="processing {} adjacency file".format(name))]
    if 'node_attributes' in dataset:
        node_features = [process_line(x) for x in tqdm(open(dataset['node_attributes']).readlines(),desc="processing {} node attribute file".format(name))]
    elif 'node_labels' in dataset:
        node_features = [process_line(x) for x in tqdm(open(dataset['node_labels']).readlines(),desc="processing {} node label file".format(name))]
    else:
        raise Exception("NONONONONONO")

    labels = [process_line(x) for x in tqdm(open(dataset['graph_labels']).readlines(),desc="processing {} graph label file".format(name))]
    graph_indicator = [process_line(x) for x in tqdm(open(dataset['graph_indicator']).readlines(),desc="processing {} graph indicator".format(name))]
    
    # import numpy as np
    # A = np.array(A)
    # print(np.max(A))

    graphs = dict()
    num_nodes = dict()
    for node1,node2 in tqdm(A,desc="generating {} data dictionary".format(name)):
        node1 = node1 - 1
        node2 = node2 - 1
        assert graph_indicator[node1] == graph_indicator[node2]
        graph_index = graph_indicator[node1] - 1
        if graph_index >= len(graphs):
            label = labels[graph_index]
            graphs[graph_index] = {'adjacency':list(),'node_features':dict(),'label':label}
            num_nodes[graph_index] = {'max':-1,'min':1e10}

        graphs[graph_index]['adjacency'].append([node1,node2])
        graphs[graph_index]['adjacency'].append([node2,node1])
        graphs[graph_index]['node_features'][node1] = node_features[node1]
        graphs[graph_index]['node_features'][node2] = node_features[node2]
        if node1 > num_nodes[graph_index]['max']:
            num_nodes[graph_index]['max'] = node1
        if node2 > num_nodes[graph_index]['max']:
            num_nodes[graph_index]['max'] = node2
        if node1 < num_nodes[graph_index]['min']:
            num_nodes[graph_index]['min'] = node1
        if node2 < num_nodes[graph_index]['min']:
            num_nodes[graph_index]['min'] = node2
        

    assert len(graphs) == len(labels)
    num_nodes = max([x['max']-x['min'] for _,x in num_nodes.items()])
    print(num_nodes)
    for index,graph in graphs.items():
        nodes = np.array(graph['adjacency'])
        min_id = np.min(nodes)
        
        feature_matrix = np.zeros()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for train/val split')
    # parser.add_argument('--load_path', type=str, required=True,
    #                     help='path to raw data directory')
    # parser.add_argument('--save_path', type=str, required=True,
    #                     help='path to save processed data')

    args = parser.parse_args()
    args.load_path = 'data/DD'
    # one TFRecord = {'adjacency':NxN matrix, 'node features':NxF matrix,'label':one_hot}
    files = glob(os.path.join(args.load_path,"**","*.txt"),recursive=True)
    datasets = get_data_dictionary(files)
    for k,v in datasets.items():
        process_data(v,k)


