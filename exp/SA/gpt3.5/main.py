import re
import os
import sys
import json
import math
import torch
import random
import psutil
import pickle
import tiktoken
import fasttext
import warnings
import argparse
import numpy as np
import pandas as pd
from io import StringIO
from collections import Counter
from cdr.message import Message, MessageRole
from sklearn.exceptions import ConvergenceWarning
from connector import GPTConnector as Connector
from serializer import DFSerializer as Serializer


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


def _clean(self, s: str) -> str:
    pattern = re.compile(r"[\|\\\-<>\/]")
    return pattern.sub("", s)

def sa_serialize(df, cell_sep = ','):
    df = df.astype(str)

    str_builder = ""
    str_builder += "pd.DataFrame({\n"
    for idx, col in enumerate(df.columns):
        values = df[col].tolist()
        values = [v for v in values if v not in {'None', 'NaN', 'none', 'nan'}]
        value_counts = Counter(values)
        filtered_values = []
        for value, count in value_counts.items():
            if count > 5:
                filtered_values.extend([value] * 5)  
            else:
                filtered_values.extend([value] * count)  

        ser_col = cell_sep.join(filtered_values)
        ## for SA, include actual header
        str_builder += f"{col}: {ser_col}, \n"

    index = df.columns
    serialized_index = ", ".join(map(str, index))
    str_builder += f"Headers already used: [{serialized_index}]"
    str_builder += "})"

    return str_builder

def clean_text(cell):
    if isinstance(cell, str):
        cleaned_text = re.sub(' +', ' ', str(cell)).strip()  
        tokens_count = len(tokenizer.encode(cleaned_text))
        return cleaned_text, tokens_count
    elif isinstance(cell, list):
        cleaned_list = []
        tokens_count = 0
        for item in cell:
            cleaned_item, item_tokens = clean_text(item)
            cleaned_list.append(cleaned_item)
            tokens_count += item_tokens
        return cleaned_list, tokens_count
    elif isinstance(cell, dict):
        cleaned_dict = {}
        tokens_count = 0
        for key, value in cell.items():
            cleaned_value, value_tokens = clean_text(value)
            cleaned_dict[key] = cleaned_value
            tokens_count += value_tokens
        return cleaned_dict, tokens_count
    elif isinstance(cell, (int, float)):
        cleaned_text = str(cell)
        tokens_count = len(tokenizer.encode(cleaned_text))
        return cell, tokens_count
    else:
        cleaned_text = re.sub(' +', ' ', str(cell)).strip()
        tokens_count = len(tokenizer.encode(cleaned_text))
        return cleaned_text, tokens_count

def sa_create_dataframe(tableData, processed_tableHeaders):
    import pandas as pd
    import random

    rows = []
    for row in tableData:
        row_data = [cell['text'] if cell['text'] else None for cell in row]
        rows.append(row_data)
    dataframe = pd.DataFrame(rows, columns=processed_tableHeaders)

    # de-duplicate columns by randomly selecting one among duplicates
    def remove_duplicate_columns(df):
        columns_to_keep = {}
        for col in df.columns:
            if col not in columns_to_keep:
                indices = [i for i, x in enumerate(df.columns) if x == col]
                selected_index = random.choice(indices)
                columns_to_keep[col] = selected_index
        unique_cols = [df.columns[columns_to_keep[col]] for col in columns_to_keep]
        df = df.iloc[:, [columns_to_keep[col] for col in columns_to_keep]]
        df.columns = unique_cols  # Set the columns to their names
        return df

    dataframe = remove_duplicate_columns(dataframe)
    cleaned_data = dataframe.applymap(clean_text)
    cleaned_text_df = cleaned_data.applymap(lambda x: x[0] if isinstance(x, tuple) else x)
    token_count_df = cleaned_data.applymap(lambda x: x[1] if isinstance(x, tuple) else x)
    total_tokens = token_count_df.sum().sum()

    return cleaned_text_df

def load_raw(path):
    header_data_dict = {}
    header_data_id_list = []
    test = []
    with open(path + 'header_vocab.txt') as f:
        task_data = f.read().splitlines() 
    with open(path + 'test_headers.json') as f:
        all_header_data = json.load(f)
        for header_list in all_header_data:
            header_data_id_list.append(header_list[0])
            header_data_dict[header_list[0]] = header_list
    with open(path + 'test_tables.jsonl') as f:
        for line in f:
            j_obj = json.loads(line)
            if j_obj['_id'] in header_data_id_list:
                df = sa_create_dataframe(j_obj['tableData'], j_obj['processed_tableHeaders'])
                test.append({'id':j_obj['_id'], 'table':df})
    return test



def load_map(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    original_phrase, representative_phrase = parts
                    mapping[original_phrase] = representative_phrase
                else:
                    print(f"Warning: Line format incorrect: '{line.strip()}'")
    return mapping

def sa_message(df, label_space, ser):
    # sys_p

    ser_labels = ', '.join(label_space)

    p1 = ''
    p1 += f'''You are an expert in analyzing tabular data and recommending the most relevant NEW column attributes based on the table's content and existing headers.'''
    p1 += f'''Given the following list of potential new headers: {ser_labels}, your task is to rank these headers from most to least relevant.'''
    p1 += f'''Please reply with only the top 10 ranked headers but nothing else (e.g., do not provide explanations). Make sure all given headers exist in your response in ranked order.'''
    p1 += f'''Please provide your answer in JSON format, listing the headers in order of relevance.'''

    sys_p = [Message(MessageRole.SYS, p1)]

    ser_context = sa_serialize(df=df)
   
    ## no use caption
    p2 = f'''
        Given the following table in dataframe format: \n
        {ser_context} \n


        Your task is to:
        1. Analyze the provided table, focusing on the existing column headers, and content of the cell values.
        2. Carefully examine the table to understand the context.
        3. From the given set of possible column headers, rank all the headers in order of their relevance for augmenting the table schema and content.

        Please provide your answer in a ranked list, based on the relevance of each header to the existing data.
        '''

    user_p = [Message(MessageRole.USER, p2)]

    # json_p
    json_p=[]
    p3 = {}
    p3[f'Headers_from_most_relevant_to_least_relevant'] = 'YOUR ANSWER HERE'

    json_p.append(Message(MessageRole.USER, json.dumps(p3, indent=4)))

    return sys_p, user_p, json_p


def parse_response_rank(response, gt, topklabels):
    if not response:
        return [''] * gt
    try:
        preds = list(response['response'].values())[0]
        if not isinstance(preds, list):
            print(f'''preds not list {preds}''')
            return topklabels 
        return preds
    except:
        return topklabels 


def run(args):


    data = load_raw(args.data)
    ser = Serializer()
    ## Task-data-reducer
    reduced_labels = pickle.load(open(args.label, 'rb'))
    down_sample_map = load_map("../../../src/data/map1749.txt")

    if not args.ntables or args.ntables > len(data):
        ntables = len(data)
    else:
        ntables = args.ntables
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    conn = Connector(model_name=args.model, model_api=str(os.getenv("OPENAI_API_KEY")))
    serializer = Serializer()

    print(f'===' * 50)
    print(f'Starting SA task')
    print(f'Model: {args.model}')
    print(f'Number of tables to inference: {ntables}')
    print(f'===' * 50)



    shard_size = 500
    shards = math.ceil(ntables / shard_size)
    processed_id = set()
    for i in range(shards):
        data_shard = data[i*shard_size: (i+1)*shard_size]
        outpath = os.path.join(args.outdir, str(args.topk), f"{args.model}-shard{i}-TURL.json")
        _dir = os.path.dirname(outpath)     
        os.makedirs(_dir, exist_ok=True)

        gt_shard, pred_shard  = {}, {}
        for j in range(len(data_shard)):
            tid = data_shard[j]['id']
            if tid in processed_id:
                raise ValueError(f'Redundant tid: {tid} detected in shards-{i} and {j}th item')
            processed_id.add(tid)

            df = data_shard[j]['table']
            df = df.sort_index(axis=1)

            label_space = reduced_labels.get(int(args.topk), {}).get(tid, None)
            print(label_space)
            if not label_space: 
                continue
            label_space = label_space[0][0]
            gt = reduced_labels.get(int(args.topk), {}).get(tid, None)[0][1]

            for i, orig_label in enumerate(label_space):
                # adhoc fix
                if orig_label in down_sample_map:
                    label_space[i] = down_sample_map[orig_label]
                elif orig_label + ' ' + orig_label in down_sample_map:
                    label_space[i] = down_sample_map[orig_label + ' ' + orig_label]
                else:
                    print('no match, fall defaul')
                    label_space[i] = orig_label 
            for i, orig_label in enumerate(gt):
                gt[i] = down_sample_map[orig_label]

            ## redundant col in dataset
            for c in gt:
                if c in df.columns:
                    df.drop(c, axis=1, inplace=True)
            if len(df.columns) == 0 or len(gt) == 0:
                print("df has no columns left or no gt")
                continue
                
            # drop columns in label_space if in df already
            old_label_space = label_space
            label_space = []
            for c in old_label_space:
                if c in df.columns:
                    continue
                else:
                    label_space.append(c)

            s_msgs, p_msgs, json_msgs = sa_message(df,label_space, serializer)
                
            pred = []

            this_msg = s_msgs + p_msgs + json_msgs
            response = conn.submit(msgs=this_msg, count_tokens=False, retry=3)
            pred = parse_response_rank(response, gt, label_space)


            gt_shard[tid] = gt
            pred_shard[tid] = pred

            if j % 200 == 0 and j != 0:
                print(f'Processed {j} tables')


        # save
        with open(outpath, 'w') as f:
            res = {'gt': gt_shard, 'pred': pred_shard}
            json.dump(res, f)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SA task')
    parser.add_argument('--model', type=str, help='model name or path')
    parser.add_argument('--data', type=str, help='Path to the data file')
    parser.add_argument('--topk', type=str, help='topk val')
    parser.add_argument('--label', type=str, help='Path to the reduced SA labels')
    parser.add_argument('--outdir', type=str, help='Dir to save the output')
    parser.add_argument('--ntables', type=int, help='Number of tables to inference')
    args = parser.parse_args()
    run(args)
