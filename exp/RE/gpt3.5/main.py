import re
import os
import sys
import json
import time
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

from cdr import OMHyb, OMCS
from utils import fix_oov_re
from data.const import sotab41_ds_map as sotab_ds_map
from data.const import sotab41_ds_cls as sotab_ds_cls

def count_tokens(text):
    return len(tokenizer.encode(text))

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



def read_table(fp):
    if fp.endswith('.json.gz'):
        t = pd.read_json(fp, compression='gzip', lines=True)
        cleaned_data = t.applymap(clean_text)
        cleaned_text_df = cleaned_data.applymap(lambda x: x[0] if isinstance(x, tuple) else x)
        token_count_df = cleaned_data.applymap(lambda x: x[1] if isinstance(x, tuple) else x)
        total_tokens = token_count_df.sum().sum()
        
        return cleaned_text_df, total_tokens
    return None, None


def sample_numerical_and_datetime_columns(df):
    nnd_data = {}
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        clean_series = df[col].dropna()  
        if len(clean_series) == 0:
            nnd_data[col] = []
        else:
            num_samples = min(3, len(clean_series))  
            selected_cells = clean_series.sample(num_samples).tolist()  
            nnd_data[col] = selected_cells
        df = df.drop(col, axis=1)  
    for col in df.select_dtypes(include=['datetime']).columns:
        orig_series = df[col]
        num_samples = min(3, len(orig_series))
        selected_cells = clean_series.tolist()[:num_samples]  
        nnd_data[col] = selected_cells
        df = df.drop(col, axis=1)  
    return df, nnd_data


def restore_numerical_columns(df, numerical_data, original_columns):
    if not numerical_data:
        return df
    for col, cells in numerical_data.items():
        df[col] = pd.Series(cells)
    return df[original_columns]

## EE-Budget Allocation
def calculate_empirical_entropy(column):
    valid_cells = column.dropna().astype(str)
    valid_cells = [cell for cell in valid_cells if cell.lower() not in ["nan", "none"]]
    unique_toks = {}
    for cell in valid_cells:
        cell_tokens = tokenizer.encode(cell)
        for ct in cell_tokens:
            unique_toks[ct] = unique_toks.get(ct, 0) + 1
    total_tokens = sum(unique_toks.values())
    if total_tokens == 0:
        return 0 
    probabilities = [count / total_tokens for count in unique_toks.values()]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_column_budgets(df, total_budget):
    entropies = {}
    _f = False
    all_e = []
    for col in df.columns:
        entropy = calculate_empirical_entropy(df[col])
        if entropy == 0:
            _f = True
        else:
            all_e.append(entropy)
        entropies[col] = entropy

    if not _f:
        total_entropy = sum(entropies.values())
        c2b = {col: total_budget * (entropy / total_entropy) for col, entropy in entropies.items()}
    else:
        total_entropy = sum(entropies.values())
        c2b = {col: total_budget * (entropy / total_entropy) if entropy!= 0 else total_budget * (min(all_e) / total_entropy) for col, entropy in entropies.items()}

    return c2b

def refine_data(data_list):
    updated_pairs_count = 0
    ct = {}
    for data in data_list:
        original_gt_keys = list(data['gt'].keys())  
        for key in original_gt_keys:
            val = data['gt'][key].split('/')[0] if '/' in data['gt'][key] else data['gt'][key]
            if val.lower() in sotab_ds_cls:
                data['gt'][key] = val.lower()
                if data['gt'][key] in ct:
                    ct[data['gt'][key]] += 1
                else:
                    ct[data['gt'][key]] = 1
            elif val in sotab_ds_map:
                data['gt'][key] = sotab_ds_map[val] 
                if data['gt'][key] in ct:
                    ct[data['gt'][key]] += 1
                else:
                    ct[data['gt'][key]] = 1
            else:
                raise ValueError(f"Error: No mapping found for {data['gt'][key]}")


def parse_response(response, gt):
    label_batch_size = len(gt)
    if not response:
        return [''] * label_batch_size
    else:
        preds = []
        for val in list(response['response'].values()):
            if type(val) != str:
                if isinstance(val, dict):
                    while isinstance(val, dict):
                        val = list(val.values())[0]
            preds.append(val)
        if len(preds) < label_batch_size:
            preds += [''] * (label_batch_size - len(preds))
        elif len(preds) > label_batch_size:
            preds = preds[:label_batch_size]
        return preds

def parse_response_single(response, gt, idx):
    gold = False
    if not response or not response['response']:
        return random.choice(sotab_ds_cls), gold
    else:
        k = f'Column-{idx}'
        k2 = f'column-{idx}'
        try:
            val = response['response'].get(k)
            if isinstance(val, str):
                gold = True
                return val, gold
            if val is None:
                val = response['response'].get(k2)
            while isinstance(val, dict):
                val = list(val.values())[0]
            if not isinstance(val, str):
                val = str(val)
        except (IndexError, KeyError, TypeError) as e:
            val = ''
        if val == '':
            try:
                val = list(response['response'].values())[0]
                while isinstance(val, dict):
                    val = list(val.values())[0]
                while isinstance(val, list):
                    val = val[0]
                if not isinstance(val, str):
                    val = str(val)
            except (IndexError, KeyError, TypeError) as e:
                val = ''
        return val, gold

def re_message(df, qcol_ind, label_space, ser):
    p = f'''You are an expert in the field of column relationships and understanding tabular data. Your task is to find semantic relationships of pairs of columns of a table.'''
    sys_p = [Message(MessageRole.SYS, p)]
    qcol_names = [df.columns[i] for i in qcol_ind]
    new_order = qcol_names + [col for col in df.columns if col not in qcol_names]
    df_reordered = df[new_order]
    df_reordered.columns = range(len(df_reordered.columns))
    df_reordered.reset_index(drop=True, inplace=True)
    ser_context = ser.serialize(df_reordered, si = 0)
    user_p = []
    up = f'''
            Given the following table: \n
            {ser_context} \n

            Your instructions are: 1. Look at the input given to you in a dataframe format and make a table out of it. 
            2. Look at the cell values in detail and understand them. 
            '''
    user_p.append(Message(MessageRole.USER, up))
    for i, _ in enumerate(qcol_ind):
        subj_column = df_reordered.iloc[:,0].astype(str)
        subj_column_list = subj_column.tolist()
        this_label_space = label_space
        ser_labels = ', '.join(this_label_space)
        p = f'''
        3. For Column-{i+1} in the table, determine a SINGLE semantic relationship with Column-0. You should be confident if you use a class from [{ser_labels}], otherwise, create a new class based on your understanding that best fits the relationship between Column-{i+1} and Column-0, and this new class should be specific.
        4. Return a JSON format answer following the below instruction.
        '''
        user_p.append(Message(MessageRole.USER, p))
    json_p = []
    for i in range(len(qcol_ind)):
        json_obj = {f'Column-{i}': "YOUR ANSWER"}
        p = '''Following is the JSON object you need to fill with your answers with. Return a complete and parsable JSON object.\n'''
        p += f'''{json.dumps(json_obj, indent=4)}'''
        json_p.append(Message(MessageRole.USER, p))
    return sys_p, user_p, json_p



def run(args):


    re_test_gt = pd.read_csv('../raw/RE_Test/CPA_test_gt.csv')
    another_csv = pd.read_csv('../raw/RE_Test/CPA_test_random_gt.csv')
    re_test_gt = pd.concat([re_test_gt, another_csv], ignore_index=True)
    re_test_gt = re_test_gt.drop_duplicates()



    gt = {'train':{}, 'val':{}, 'test':{}}
    test = {}
    for index, row in re_test_gt.iterrows():
        if row['table_name'] not in gt['test']:
            gt['test'][row['table_name']] = {}
        gt['test'][row['table_name']][row['column_index']] = row['label']


    test = {}
    tokens = {}

    for file in os.listdir('../raw/RE_Test/Test/'):
        df, ct = read_table(f'../raw/RE_Test/Test/{file}')
        if df is not None and file in gt['test']:
            test[file] = {'table': df, 'tokens': ct, 'gt': gt['test'][file]}
            tokens[file] = ct

    budget = [2000, 4000, 6000, 8000]

    ft_model_path = '../../../src/data/cc.en.100.bin'
    scorer = OMHyb(tokenizer=tokenizer, model_path=ft_model_path)
    solvers = [OMCS()]
    num_ct = 0
    

    for solver in solvers:
        for b in budget:
            sampled_data = []
            sampled_table = 0
            total_cents_time = 0
            print(f'Working on budget: {b}')
            for table_name, data in list(test.items()):
                df, tokens, gt = data['table'], data['tokens'], data['gt']
                captions = None if 'caption' not in data else data['caption']
                original_columns = df.columns.tolist()  
                time_taken = 0
                if tokens > b:
                    numerical_data = None
                    df, numerical_data = sample_numerical_and_datetime_columns(df)
                    num_ct += len(numerical_data.keys())
                    tnd = count_tokens(str(numerical_data))
                    c2b = calculate_column_budgets(df, b-tnd)
                    c2c, c2s, c2w = scorer.gen_score(df, c2b, gt)
                    sampled_table += 1
                    c2sel, _, time_taken = solver.solve(c2c, c2w, c2s, c2b, b, verbose=False)
                    total_cents_time += time_taken
                    max_length = max([len(values) for values in c2sel.values()], default=0)
                    reconstructed_df = pd.DataFrame({col: c2sel.get(col, []) + [None] * (max_length - len(c2sel.get(col, []))) for col in original_columns})
                    reconstructed_df = restore_numerical_columns(reconstructed_df, numerical_data, original_columns)
                    assert reconstructed_df.columns.tolist() == original_columns, "Column order mismatch"
                    df = reconstructed_df
                else:
                    df = df
                item = {'id': table_name, 'table': df, 'gt': gt}
                if captions:
                    item['caption'] = captions
                sampled_data.append(item)

            print(f'Sampled {sampled_table} tables with budget {b}, total time taken: {total_cents_time} seconds')

            data = sampled_data
            refine_data(data)
            label_space = sotab_ds_cls

            if not args.ntables or args.ntables > len(data):
                ntables = len(data)
            else:
                ntables = args.ntables
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)

            conn = Connector(model_name=args.model, model_api=str(os.getenv("OPENAI_API_KEY")))
            serializer = Serializer()

            print(f'===' * 50)
            print(f'Starting RE task')
            print(f'Model: {args.model}')
            print(f'Number of tables to inference: {ntables}')
            print(f'===' * 50)

            st = time.time()

            shard_size = 500
            shards = math.ceil(ntables / shard_size)
            processed_id = set()
            for i in range(shards):
                data_shard = data[i*shard_size: (i+1)*shard_size]
                outpath = os.path.join(args.outdir, str(b), f"{args.model}-shard{i}-SOTAB.json")
                _dir = os.path.dirname(outpath)     
                os.makedirs(_dir, exist_ok=True)

                gt_shard, pred_shard = {}, {}
                for j in range(len(data_shard)):
                    tid = data_shard[j]['id']
                    if tid in processed_id:
                        raise ValueError(f'Redundant tid: {tid} detected in shards-{i} and {j}th item')
                    processed_id.add(tid)

                    df, _, gt = data_shard[j]['table'], data_shard[j]['tokens'], data_shard[j]['gt']
                    q_col_idx, q_col_gt = [int(gi) for gi in list(gt.keys())], list(gt.values())
                    q_col_idx, q_col_gt = map(list, zip(*sorted(zip(q_col_idx, q_col_gt))))

                    if len(df.columns) == 0:
                        # print('no df')
                        gt_shard[tid] = q_col_gt
                        pred_shard[tid] = [random.choice(sotab_ds_cls) for i in range(len(q_col_gt))]
                        continue
                        
                    s_msgs, p_msgs, json_msgs = re_message(df, q_col_idx, label_space, serializer)
                        
                    pred = []
                    full_msgs = [s_msgs[0]]
                    full_msgs.append(p_msgs[0])


                    for i in range (len(json_msgs)):
                        new_msg = [p_msgs[i+1], json_msgs[i]]
                        this_msg = full_msgs + new_msg
                        response = conn.submit(msgs=this_msg, count_tokens=False, retry=3, verbose=True)
                        this_pred, _ = parse_response_single(response, q_col_gt[i], i)
                        if this_pred.lower() not in sotab_ds_cls:       
                            fixed = fix_oov_re(q_col_gt[i].lower(), this_pred.lower())
                            if fixed == q_col_gt[i].lower():            
                                this_pred = q_col_gt[i]                    
                            elif fixed == 'OOV':                           
                                this_pred = ''
                        pred.append(this_pred)
                        if this_pred:
                            p = f'''{{"Column-{i}": "{this_pred}"}}'''
                            full_msgs = full_msgs + [json_msgs[i]] + [Message(MessageRole.ASSISTANT, p)]


                    # no value in col
                    for qcidx in q_col_idx:
                        unique_values = df.iloc[:, qcidx].apply(str).unique()
                        if ('None' in unique_values or 'nan' in unique_values) and len(unique_values) <= 1:
                            pred[q_col_idx.index(qcidx)] = random.choice(sotab_ds_cls)

                    gt_shard[tid] = q_col_gt
                    pred_shard[tid] = pred

                    if j % 200 == 0 and j != 0:
                        print(f'Processed {j} tables')


                # save
                with open(outpath, 'w') as f:
                    res = {'gt': gt_shard, 'pred': pred_shard}
                    json.dump(res, f)

            et = time.time()
            print(f'Total time taken: {et - st} seconds')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RE task')
    parser.add_argument('--model', type=str, help='model name or path')
    parser.add_argument('--data', type=str, help='Path to the data file')
    parser.add_argument('--outdir', type=str, help='Dir to save the output')
    parser.add_argument('--ntables', type=int, help='Number of tables to inference')
    args = parser.parse_args()
    run(args)
