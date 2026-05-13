import sys
from openai import OpenAI
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
import tiktoken
from tqdm import tqdm
start_time = time.time()

client = OpenAI()



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
    # fname = 'result_v1.0/llama_output_retrieval/' + event_type + '.json'
    fname = args.pred_file + event_type + '.json'
    try:
        with open(fname, 'r') as file:
            data = json.load(file)
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

def predict(doc, prompt, question):
    content = prompt + doc + question
    messages = [
        {"role": "user", "content": content}, ]
    success = 0
    attempts = 0
    while (not success) and (attempts < 5):
        # time.sleep(1)
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                # model = "gpt-4o",
                temperature=args.temperature,
                # temperature=1,
                max_tokens=args.max_length_cot,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                messages=messages,
            )
            result = completion.choices[0].message.content
            success = True
        except:
            success = False
            print(f"candidate llm-based episode failed! attempt #{attempts + 1}")
            attempts += 1
    response_text = result
    table_str = ""
    pred_table = ""
    if response_text is not None:
        def split_by_substrings(main_string, substring1, substring2):
            parts = main_string.split(substring1)
            split_parts = [part.split(substring2) for part in parts]
            flattened_parts = [item for sublist in split_parts for item in sublist]
            flattened_parts = [i.strip('\n') for i in flattened_parts if i]
            return flattened_parts

        if 'Table:' in response_text:
            substr1, substr2 = 'Explanation:', 'Table:'
            table_str = split_by_substrings(response_text, substr1,
                                            substr2)  #
        elif 'table' in response_text:
            substr1, substr2 = 'Explanation:', 'table:'
            table_str = split_by_substrings(response_text, substr1, substr2)

        if len(table_str) == 3:
            table_dic = markdown2dic(table_str[2])
            pred_table = table_str[2]
        elif len(table_str) == 2:
            table_dic = markdown2dic(table_str[1])
            pred_table = table_str[1]
    else:
        table_dic = {}
    return pred_table
def markdown2dic(input_str):
    # print(input_str)
    rows = input_str.split("\n")
    rows = [row.strip() for row in rows if row]
    if len(rows) == 3:
        keys = [key.strip() for key in rows[0].split("|") if key]
        values = [value.strip() for value in rows[2].split("|") if value]
        table_dict = dict(zip(keys, values))
    else:
        table_dict = {}
        for row in rows[2:]:
            values = [value.strip() for value in row.split("|") if value]
            if len(values) != 0:
                key = values[0]
                table_dict[key] = values[-1]
    return table_dict
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens
def iter_cluster_demos(text):
    rcd = -1
    rcd_max_n = -1

    for nc in range(10, 2 - 1, -1):
        total_num_token = 0
        token_limit = 4096
        token_reserved = 512
        fname = args.pred_file + '_' + str(nc) + '/' + event_type + '.json'
        print(fname)
        with open(fname, 'r') as file:
            dem_arguments = json.load(file)

            for line in dem_arguments:
                total_num_token += num_tokens_from_string(line["pred_ans"])
                # total_num_token += num_tokens_from_string(line["doc"])  ###计算当前输入的所有检索样例token数量

        if total_num_token > token_limit - token_reserved:
            continue
        else:
            if len(dem_arguments) > rcd_max_n:
                rcd_max_n = len(dem_arguments)
                rcd = nc

    print("rcd: " + str(rcd), "rcd_max_n: " + str(rcd_max_n))###打印最大能输入的演示样本数量
    while True:
        try:
            print("---------------rcd: " + str(rcd) + "---------------")
            if rcd == -1:
                break
            x = []
            z = []

            fname = args.pred_file + '_' + str(rcd) + '/' + event_type + '.json'

            with open(fname, 'r') as file:
                dem_arguments = json.load(file)
            for line in dem_arguments:
                x.append(line['pred_ans'])
                z.append(line['doc'])

            iters = args.iter


            print(" number of example:" + str(len(x)) + " iters: " + str(iters))

            ###迭代检索样例
            index_list = list(range(len(x)))
            for p in range(args.iter):
                index_list_new = index_list[:]
                for q in tqdm(range(len(x)), total=len(x)):
                    i = random.choice(index_list_new)
                    index_list_new.remove(i)
                    remaining_list = index_list[:i] + index_list[i + 1:]
                    # shuffle remaining_list
                    remaining_list_copy = remaining_list[:]
                    random.shuffle(remaining_list_copy)
                    demo_text = ""
                    id = 0
                    for r in remaining_list_copy:
                        demo_text += "Example %d:\n\nTable:\n%s\n\n\n" % (id +1, x[r])
                        id += 1
                    prompt = "Given an event trigger and the background text, extract the information as a markdown table and produce a paragraph as the explanation. \
                                       Let this table include as many columns as possible and ensure the content is brief. \n\n \
                                       Please follow the below output format:\n \
                                       Explanation:\n \
                                       [a paragraph of what information should be extracted and why]\n \
                                       Table:\n \
                                       [a markdown table following the above explanation] \n\n \
                                       Here are several examples: \n\n "

                    prompt += demo_text
                    prompt += 'Please adopt a step-by-step approach: generate a comprehensive explanation as the first step, followed by table extraction as the second step:'

                    x_new = predict(z[i], prompt, question)
                    x[i] = x_new
            break
        except Exception as e:
            print(str(rcd) + " error" + str(e))
            rcd -= 1
            print("---------------rcd: " + str(rcd) + "---------------")

    demos=[]
    for i in range(len(dem_arguments)):
        demo_element = {
            "doc": z[i],
            "pred_ans": x[i],
        }
        demos.append(demo_element)
    dem_path = args.demo_save_dir + '_' + str(rcd) + '_' + str(args.iter)
    if not os.path.exists(dem_path):
        os.makedirs(dem_path)
    path = dem_path + '/' + event_type + '.json'
    with open(path, 'w', encoding="utf-8") as file:
        file.write(json.dumps(demos, indent=4))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_file",
        type=str,
        default='cluster_demos/gpt-3.5-turbo',
    )
    parser.add_argument(
        "--demo_save_dir",
        type=str,
        default="iter_demos/gpt-3.5-turbo",
        help="where to save the contructed demonstrations"
    )
    parser.add_argument("--random_seed",
                        type=int,
                        default=192,
                        help="random seed")

    parser.add_argument(
        "--num_clusters",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=5,
        help="iterations of CAT"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="temperature for gpt-3.5-turbo"
    )
    parser.add_argument(
        "--max_length_cot",
        type=int,
        default=512,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    fix_seed(args.random_seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # encoder = SentenceTransformer(args.encoder)
    DATA_DIR = './RoleEE_data/'  # dataset directory
    # OUTPUT_DIR = 'result_v1.0/llama_similarity_filter_output/'
    EVENT_TYPE = []
    for root, dirs, files in os.walk(DATA_DIR, topdown=False):
        EVENT_TYPE = dirs
    # EVENT_TYPE = ['academy award ceremony']
    for event_type in EVENT_TYPE:
        path = DATA_DIR + event_type + '/'

        # Load multiple documents for the given event type
        text, triggers = load_data(path)
        # Truncate each document
        text = truncate_text(text)
        # dem_arguments = load_events(event_type)
        ###Iterative retrieval
        question = ' According to this, what is the key information of this ' + event_type + '?'
        demos = iter_cluster_demos(text)

        # save_dem_data(event_type, demos)
