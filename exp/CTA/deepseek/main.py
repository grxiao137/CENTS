# %%
## set cache dir for hf
import os
import re
import torch
import json
import math
import random
import warnings
import psutil
import pickle
import argparse
import numpy as np
import pandas as pd
from io import StringIO
from collections import Counter
from openai import RateLimitError, APIError
from sklearn.exceptions import ConvergenceWarning

from utils import *
from data.const import sotab27_ds_map as sotab_ds_map


from cdr import OMHyb, OMCS
from utils import fix_oov_cta
from serializer import DFSerializer as Serializer
from data.const import sotab27_ds_map as sotab_ds_map
from data.const import sotab27_ds_cls as sotab_ds_cls



warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

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

def _clean(s: str) -> str:
    pattern = re.compile(r"[\|\\\-<>\/]")
    return pattern.sub("", s)

def refine_data(data_list):
    initial_items_count = len(data_list)
    initial_pairs_count = sum(len(item['gt']) for item in data_list)
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



def find_in_labels(response, sotab_ds_cls):
    for cls in sotab_ds_cls:
        if cls in response:
            return cls
    return None

def parse_response_single_find(response, idx, sotab_ds_cls):
    k = f'Column-{idx}'
    k2 = f'column-{idx}'
    
    start = response.find('{')
    end = response.rfind('}')
    
    json_str = None
    if start != -1 and end != -1 and start < end:
        json_str = response[start:end+1]

    if json_str:
        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            parsed = None
    else:
        parsed = None

    if parsed is None:
        cls = find_in_labels(response, sotab_ds_cls)
        if not cls:
            ## randomly choose one
            print(f'fail to parse response: {response}, returning random')
            return random.choice(sotab_ds_cls)
        else:
            return cls
    else:
        k = f'Column-{idx}'
        k2 = f'column-{idx}'
        try:
            val = parsed.get(k)
            if isinstance(val, str):
                return val
    
            if val is None:
                val = parsed.get(k2)
    
            while isinstance(val, dict):
                val = list(val.values())[0]
            if not isinstance(val, str):
                val = str(val)
        except (IndexError, KeyError, TypeError):
            cls = find_in_labels(response, sotab_ds_cls)
            if not cls:
                ## randomly choose one
                print(f'fail to parse response: {response}, returning random')
                return random.choice(sotab_ds_cls)
            else:
                return cls
    
    return val


def df_serialize(df, cell_sep = ',', si = 0):
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
            ## for others
            str_builder += f"Column-{si + idx}: {ser_col}, \n"


        index = df.columns
        serialized_index = ", ".join(map(str, index))
        str_builder += f"Index: [{serialized_index}]"
        str_builder += "})"

        return str_builder


def gen_prompt(df, qcol_ind, gt_labels, labels, ser=df_serialize):
        qcol_names = [df.columns[i] for i in qcol_ind]
        new_order = qcol_names + [col for col in df.columns if col not in qcol_names]

        df_reordered = df[new_order]
        df_reordered.columns = range(len(df_reordered.columns))
        df_reordered.reset_index(drop=True, inplace=True)
        # print('df_reordered', df_reordered)

        ser_context = ser(df_reordered)

        context_p = []
        json_p = []

        assert len(gt_labels) == len(qcol_ind), "Length of gt_labels must be equal to the number of columns to annotate"

        tab_p = f'''
            Given the following table: \n
            {ser_context} \n
            Your instructions are: 1. Look at the input given to you in a dataframe and understand it.
        '''
        this_label_space = labels
        ser_labels = ', '.join(this_label_space)

        for i, gt in enumerate(gt_labels):
                this_column = df_reordered.iloc[:,i].astype(str)
                this_column_list = this_column.tolist()
                values = [_clean(v) for v in this_column_list if v not in {'None', 'NaN', 'none', 'nan'}]
                value_counts = Counter(values)
                filtered_values = []
                for value, count in value_counts.items():
                    if count > 3:
                        filtered_values.extend([value] * 3)  
                    else:
                        filtered_values.extend([value] * count)  

                ser_column = ', '.join(filtered_values)
                p = f'''
                2. For Column-{i} in the above table, based on Column-{i}'s content and other columns' context, assign a semantic class to the best you can to it that best represents all cells of Column-{i} from  [{ser_labels}].\n
                3. Return a JSON format answer following the below instructions. \n
                '''
                context_p.append(p)

        for i in range(len(gt_labels)):
                json_obj = {f'Column-{i}': "YOUR ANSWER"}
                p = '''Following is the JSON object you need to fill with your answers with. After your reasoning ending with </think>, directly return a complete and parsable JSON object.\n'''
                p += f'''{json.dumps(json_obj, indent=4)}'''
                ## For reasoning model
                p += f'''\n<think>'''
                json_p.append(p)

        return tab_p, context_p, json_p


# %%
def gen_response(model, tokenizer, prompt, max_new_tokens, parse_thinking=True, verbose=False):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
    if verbose:
        print(f'Generated_text (with reasoning): {generated_text}')
        
    if parse_thinking:
        marker = '</think>'
        marker_index = generated_text.find(marker)
        generated_text = generated_text[marker_index + len(marker):]
        if verbose:
            print(f'Generated_text (w/o reasoning): {generated_text}')
    return generated_text



def run(args):

    # %%
    if torch.cuda.is_available():
        print("GPU is available.")
        print("Number of GPUs:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("GPU is not available.")



    cache_dir = './cache/'

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        device_map="auto",
    )
    ds_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir)
    ds_tokenizer.pad_token_id = ds_tokenizer.eos_token_id

    cta_test_gt = pd.read_csv('../raw/CTA_Test/CTA_test_gt.csv')

    gt = {'train':{}, 'val':{}, 'test':{}}
    test = {}
    for index, row in cta_test_gt.iterrows():
        if row['table_name'] not in gt['test']:
            gt['test'][row['table_name']] = {}
        gt['test'][row['table_name']][row['column_index']] = row['label']


    test = {}
    tokens = {}

    for file in os.listdir('../raw/CTA_Test/Test/'):
        df, ct = read_table(f'../raw/CTA_Test/Test/{file}')
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

            data = sampled_data
            refine_data(data)
            label_space = sotab_ds_cls

            if not args.ntables or args.ntables > len(data):
                ntables = len(data)
            else:
                ntables = args.ntables
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)

            print(f'===' * 50)
            print(f'Starting CTA task')
            print(f'Number of tables to inference: {ntables}')
            print(f'===' * 50)

            shard_size = 500
            shards = math.ceil(ntables / shard_size)
            processed_id = set()

            for i in range(shards):
                data_shard = data[i*shard_size: (i+1)*shard_size]

                outpath = os.path.join(args.outdir, str(b), f"shard{i}-SOTAB.json")
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
                        
                    tab_p, c_p, j_p = gen_prompt(df=df, qcol_ind=q_col_idx, gt_labels= q_col_gt, labels=label_space, ser=df_serialize)
                        
                    pred = []

                    history = [tab_p]

                    for i in range (len(j_p)):
                        new_msg = [c_p[i], j_p[i]]
                        this_msg = history + new_msg
                        this_msg_str = '\n'.join(this_msg)
                        response = gen_response(model, tokenizer, this_msg_str, 4096, parse_thinking=True, verbose=False)
                        this_pred = parse_response_single_find(response, i, sotab_ds_cls)
                        if this_pred.lower() not in sotab_ds_cls:       
                            fixed = fix_oov_cta(q_col_gt[i].lower(), this_pred.lower())
                            if fixed == q_col_gt[i].lower():            
                                this_pred = q_col_gt[i]                    
                            elif fixed == 'OOV':                           
                                this_pred = ''
                        pred.append(this_pred)
                        if this_pred:
                            p = f'''{{"Column-{i}": "{this_pred}"}}'''
                    for qcidx in q_col_idx:
                        unique_values = df.iloc[:, qcidx].apply(str).unique()
                        if ('None' in unique_values or 'nan' in unique_values) and len(unique_values) <= 1:
                            print('unique_val:', unique_values)
                            pred[q_col_idx.index(qcidx)] = random.choice(sotab_ds_cls)

                    gt_shard[tid] = gt
                    pred_shard[tid] = pred

                print(f'saving to {outpath}')
                with open(outpath, 'w') as f:
                    res = {'gt': gt_shard, 'pred': pred_shard}
                    json.dump(res, f)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deepseek cta')
    parser.add_argument('--data', type=str, help='Path to the data file')
    parser.add_argument('--outdir', type=str, help='Dir to save the output')
    parser.add_argument('--ntables', type=int, help='Number of tables to inference')
    args = parser.parse_args()
    run(args)
