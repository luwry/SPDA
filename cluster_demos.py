import sys
from nltk import sent_tokenize, word_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from sklearn.model_selection import train_test_split
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import csv
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import json
import re
sim_model = SentenceTransformer("./all-MiniLM-L6-v2")
import time
import random
import argparse
start_time = time.time()
def load_data(path):
    # Load multiple documents from the path.
    # Each document describes one event instance.
    data = []
    trigger = []
    file_names = os.listdir(path)
    for name in file_names:
        if '.txt' in name:
            f = open(path + name, 'r', encoding="utf-8")
            lines = f.readlines()
            text = ' '.join(lines)
            data.append(text)
            # The name of each document is regarded as the event trigger.
            # Note that the triggers are only used to identify different events. They won't be used for role prediction.
            trigger.append(name[:-4])
    return data, trigger
def truncate_text(text, max_length=350):
    # Truncate each document for subsequent processing
    # Note that the length of the truncated text here should be less than the maximum length that pretrained models can process.
    # Because the prompt will be added to generate candidate roles
    truncated = []
    for doc in text:
        string = []
        cnt_w = 0
        sents = sent_tokenize(doc)
        for s in sents:
            c = len(word_tokenize(s))
            if cnt_w + c > max_length:
                break
            cnt_w += c
            string.append(s)
        truncated.append(string)
    return truncated
def load_events(event_type):
    fname = args.pred_file + event_type + '.json'
    print(fname)
    try:
        with open(fname, 'r') as file:
            data = json.load(file)
            # print(data)
    except FileNotFoundError:
        data = []
    return data
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
def save_dem_data(event_type, data):
    dem_path = args.demo_save_dir
    if not os.path.exists(dem_path):
        os.makedirs(dem_path)
    path = dem_path + event_type + '.json'
    with open(path, 'w', encoding="utf-8") as file:
        file.write(json.dumps(data, indent=4))
def retrieval_cluster_demos(text, dem_arguments, triggers):
    ####cluster retrieval
    retrieval_text = []
    doc = []
    for idx, arg in enumerate(dem_arguments):
        retrieval_text.append(arg['pred_table'])
        doc.append(arg['doc'])

    # print(len(retrieval_text))
    if len(retrieval_text) >= args.num_clusters:

        corpus_embeddings = sim_model.encode(retrieval_text, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

        corpus_embeddings = corpus_embeddings.cpu().numpy()
        num_clusters = args.num_clusters
        clustering_model = KMeans(n_clusters=num_clusters, random_state=192)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        # print("cluster label")
        # print(cluster_assignment)
        clustered_sentences = [[] for i in range(num_clusters)]

        dist = clustering_model.transform(corpus_embeddings)
        clustered_dists = [[] for i in range(num_clusters)]
        clustered_idx = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id].append(retrieval_text[sentence_id])
            clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
            clustered_idx[cluster_id].append(sentence_id)

        demos = []

        for i in range(len(clustered_dists)):###遍历每一个簇
            # print("Cluster ", i + 1)
            tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
            top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
            # if not args.sampling == "center":
            #     random.shuffle(top_min_dist)
            for element in top_min_dist:
                min_idx = element[0]

                c_pred_ans = retrieval_text[clustered_idx[i][min_idx]]

                demo_element = {
                    "doc": doc[clustered_idx[i][min_idx]],
                    "pred_ans": c_pred_ans,
                }
                demos.append(demo_element)
                # iter_demos.append(c_pred_ans)

                break

        return demos

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_file",
        type=str,
        default='demos/gpt-3.5-turbo_zero_shot_cot/',
    )
    parser.add_argument(
        "--demo_save_dir",
        type=str,
        default="cluster_demos/gpt-3.5-turbo_5/",
        help="where to save the contructed demonstrations"
    )
    parser.add_argument("--random_seed",
                        type=int,
                        default=192,
                        help="random seed")

    parser.add_argument(
        "--num_clusters",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--retrieval_size",
        type=float,
        default=1.0,
    )
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    fix_seed(args.random_seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    DATA_DIR = './RoleEE_data/'  # dataset directory
    EVENT_TYPE = []
    for root, dirs, files in os.walk(DATA_DIR, topdown=False):
        EVENT_TYPE = dirs
    for event_type in EVENT_TYPE:
        path = DATA_DIR + event_type + '/'
        # Load multiple documents for the given event type
        text, triggers = load_data(path)
        # Truncate each document
        text = truncate_text(text)
        dem_arguments = load_events(event_type)
        # print(dem_arguments)
        ###Clustering-based retrieval
        demos = retrieval_cluster_demos(text, dem_arguments, triggers)

        save_dem_data(event_type, demos)
