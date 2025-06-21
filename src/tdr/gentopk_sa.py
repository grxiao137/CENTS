'''
Disclaimer: This code is modification based on DODUO's codebase to support generation of top-k for SA
'''
from collections import defaultdict
import json
import math
import os
import sys

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import f1_score, precision_recall_fscore_support

import ast
import torch
import pickle 
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from dataset import (collate_fn, TURLColTypeColwiseDataset,SchAugTablewiseDataset,
                    collate_fn_sotabfull,
                    SOTABFullColTypeTablewiseDataset,
                    SOTABFullRelExtTablewiseDataset,
                    TURLColTypeTablewiseDataset, TURLRelExtColwiseDataset,
                    TURLRelExtTablewiseDataset, SatoCVColwiseDataset,
                    SatoCVTablewiseDataset)

from model import BertForMultiOutputClassification, BertMultiPairPooler
from util import parse_tagname, f1_score_multilabel

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============
    #tag_name = "model/sato_bert_bert-base-uncased-bs32-ml-8"
    print(sys.argv)
    tag_name = sys.argv[1]
    folder = sys.argv[2]
    topk_list = ast.literal_eval(sys.argv[3]) 
    multicol_only = False
    shortcut_name, _, max_length = parse_tagname(tag_name)

    colpair = False

    # Single-column or multi-column
    if os.path.basename(tag_name).split("_")[1] == "single":
        single_col = True
    else:
        single_col = False

    # Task names
    task = sys.argv[7]
    #task = os.path.basename(tag_name).split("_")[0]
    num_classes_list = []
    if task == "turl":
        tasks = ["turl"]
        num_classes_list.append(int(sys.argv[4]))
    elif task == "turl-re" or task == "turl-re-colpair":  # turl-re or turl-re-colpair
        tasks = ["turl-re"]
        num_classes_list.append(int(sys.argv[4]))
        if task == "turl-re-colpair":
            colpair = True
    elif task in [
            "sato0", "sato1", "sato2", "sato3", "sato4", "msato0", "msato1",
            "msato2", "msato3", "msato4"
    ]:
        if task[0] == "m":
            multicol_only = True
        tasks = [task]  # sato, sato0 , ...
        num_classes_list.append(78)
    elif task == "turlturl-re":
        tasks = ["turl", "turl-re"]
        num_classes_list.append(int(sys.argv[4]))
        num_classes_list.append(int(sys.argv[5]))
    elif task == "turlturl-re-colpair":
        tasks = ["turl", "turl-re"]
        num_classes_list.append(int(sys.argv[4]))
        num_classes_list.append(int(sys.argv[5]))
        colpair = True
    elif task == "satoturl":
        tasks = ["sato", "turl"]
        num_classes_list.append(78)
        num_classes_list.append(121)
    elif task == "satoturlturl-re":
        tasks = ["sato", "turl", "turl-re"]
        num_classes_list.append(78)
        num_classes_list.append(121)
        num_classes_list.append(255)
    elif task == "sotabfull":
        #TODO: added by GX,
        # num_classes_list.append(82)
        tasks = ["sotabfull"]
        num_classes_list.append(27)
    elif task == "sotabfull-re":
        colpair = True
        num_classes_list.append(108)
    elif task == "sa":
        tasks = ["sa"]
        # 5652
        num_classes_list.append(int(sys.argv[6]))
    elif task == "sotabfull-all":
        tasks = ["sotabfull","sotabfull-re"]
        colpair = True
        num_classes_list.append(82)
        num_classes_list.append(108)
    else:
        raise ValueError("Invalid task name(s): {}".format(tag_name))

    print(f"Device: {device}, Tag Name: {tag_name}, Folder: {folder}, TopK List: {topk_list}, "
          f"Multi-Column Only: {multicol_only}, Shortcut Name: {shortcut_name}, Max Length: {max_length}, "
          f"Column Pairing: {colpair}, Single Column: {single_col}, Tasks: {tasks}, "
          f"Number of Classes List: {num_classes_list}")

    for task, num_classes in zip(tasks, num_classes_list):
        #output_filepath = "{}.json".format(tag_name.replace("model/", "eval/"))
        output_filepath = "{}={}.json".format(
            tag_name.replace("model/", "eval/"), task)
        output_dirpath = os.path.dirname(output_filepath)
        if not os.path.exists(output_dirpath):
            print("{} not exist. Created.".format(output_dirpath))
            os.makedirs(output_dirpath)

        #max_length = int(tag_name.split("-")[-1])
        #batch_size = 32
        batch_size = 16
        if len(tasks) == 1:
            f1_macro_model_path = "{}_best_macro_f1.pt".format(tag_name)
            f1_micro_model_path = "{}_best_micro_f1.pt".format(tag_name)
        else:
            f1_macro_model_path = "{}={}_best_macro_f1.pt".format(
                tag_name, task)
            f1_micro_model_path = "{}={}_best_micro_f1.pt".format(
                tag_name, task)
        # ============

        tokenizer = BertTokenizer.from_pretrained(shortcut_name)

        # WIP
        if single_col:
            model_config = BertConfig.from_pretrained(shortcut_name,
                                                      num_labels=num_classes)
            model = BertForSequenceClassification(model_config).to(device)
        else:
            model = BertForMultiOutputClassification.from_pretrained(
                shortcut_name,
                num_labels=num_classes,
                output_attentions=False,
                output_hidden_states=False,
            ).to(device)

        if task == "turl-re" or task == "sotabfull-re" and colpair:
            print("Use column-pair pooling")
            # Use column pair embeddings
            config = BertConfig.from_pretrained(shortcut_name)
            model.bert.pooler = BertMultiPairPooler(config).to(device)

        if  "sa" in task:
            filepath = f"data/{folder}table_sche_aug_serialized_turl_full.pkl"
            dataset_cls = SchAugTablewiseDataset
            print('loading sa')

            test_dataset = dataset_cls(filepath=filepath,
                                       split="test",
                                       folder=folder,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       multicol_only=False,
                                       device=device)
            #print(f'task is {task} with num_classes {num_classes} with total number of table {len(test_dataset)}')
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         collate_fn=collate_fn)

        elif "sotabfull" in task:
            if task in ["sotabfull"]:
                filepath = f"data/{folder}table_col_type_serialized_sotab_full.pkl"
            elif "sotabfull-re" in task:  # turl-re-colpair
                filepath = f"data/{folder}table_rel_extraction_serialized_sotab_full.pkl"
            else:
                raise ValueError("sotabfull tasks must be sotabfull or sotabfull-re.")

            if single_col:
                raise ValueError("single col not supporeted")
            else:
                if task == "sotabfull":
                    dataset_cls = SOTABFullColTypeTablewiseDataset
                elif task == "sotabfull-re":
                    dataset_cls = SOTABFullRelExtTablewiseDataset
                else:
                    raise ValueError()

            test_dataset = dataset_cls(filepath=filepath,
                                       split="test",
                                       folder=folder,
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       multicol_only=False,
                                       device=device)
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         collate_fn=collate_fn_sotabfull)
            print(f'task is {task} with num_classes {num_classes} with total number of table {len(test_dataset)}')
        else:
            raise ValueError()

        eval_dict = defaultdict(dict)
        for f1_name, model_path in [("f1_macro", f1_macro_model_path),
                                    ("f1_micro", f1_micro_model_path)]:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            ts_pred_list = []
            ts_true_list = []
            hits_at_k = {k: 0 for k in topk_list}

            # For SA
            recall_at_k = {k: 0 for k in topk_list}
            precision_at_k = {k: 0 for k in topk_list}
            f1_score_at_k = {k: 0 for k in topk_list}

            total_instances = 0
            # Test
            
            top_k_labels = {k: {} for k in topk_list}

            for batch_idx, batch in enumerate(test_dataloader):
                tids = batch["table_id"]
                if single_col:
                    # Single-column
                    raise ValueError("single-column not supported")
                else:
                    # Multi-column
                    logits, = model(batch["data"].T)
                    if len(logits.shape) == 2:
                        logits = logits.unsqueeze(0)
                    cls_indexes = torch.nonzero(
                        batch["data"].T == tokenizer.cls_token_id)
                    filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                  logits.shape[2]).to(device)
                    for n in range(cls_indexes.shape[0]):
                        i, j = cls_indexes[n]
                        logit_n = logits[i, j, :]
                        filtered_logits[n] = logit_n
                    if "sato" in task:
                        raise ValueError("sato not supported")
                    elif "sa" in task:
                        l2v = pickle.load(open('./turl5652/label_to_vec.pkl', 'rb'))
                        v2l = {str(v.index(1)): k for k, v in l2v.items()}

                        filtered_logits = torch.zeros(batch["label"].shape[0],
                                                      logits.shape[2]).to(device)

                        if len(logits.shape) == 2:
                            logits = logits.unsqueeze(0)

                        sep_indexes = torch.nonzero(
                            batch["data"].T == tokenizer.sep_token_id)

                        for n in range(sep_indexes.shape[0]):
                            i, j = sep_indexes[n]
                            logit_n = logits[i, j, :]
                            # Use `i` instead because we only need the last [SEP] logits
                            filtered_logits[i] = logit_n

                        all_labels = batch["label"].cpu().detach().numpy()

                        probabilities = torch.sigmoid(filtered_logits)

                        true_indices = torch.where(batch["label"] == 1)[1].reshape(-1, 1)
                        total_instances += true_indices.shape[0]
                        true_indices = torch.where(batch["label"] == 1)
                        # FOR SA
                        # num_relevant = true_indices.shape[0]

                        for topk in topk_list:
                            _, top_k_indices = torch.topk(probabilities, topk, dim=1)
                            # Create a tensor to keep track of hits per batch item
                            # hits_tensor = torch.zeros(top_k_indices.shape[0], dtype=torch.int)

                            # Check each batch item
                            print(f'''top_k_indices.shape: {top_k_indices.shape}''')
                            for i in range(top_k_indices.shape[0]):
                                tid = batch['table_id'][i]
                                # Get true labels for the current batch item
                                batch_true_indices = true_indices[1][true_indices[0] == i]
                                '''
                                print('this b_t_id')
                                print(batch_true_indices)
                                print('this hits')
                                print(hits_ct)
                                '''
                                hits_ct = (top_k_indices[i, :].unsqueeze(1) == batch_true_indices).any(0).sum().int()
                                hits_at_k[topk] += hits_ct

                                ## Added Sep 20 for saving topk
                                # print(batch_true_indices.cpu().numpy().shape)
                                true_indices_list = batch_true_indices.cpu().numpy().tolist()
                                top_k_list = [l[0] for l in top_k_indices[i, :].unsqueeze(1).cpu().numpy().tolist()]
                                # print(f'''true_indices_list: {true_indices_list}''')
                                # print(f'''top_k_list: {top_k_list}''')
                                true_label = [v2l[str(vi)] for vi in true_indices_list]
                                topk_label = [v2l[str(vi)] for vi in top_k_list]
                                if tid not in top_k_labels[topk]:
                                    top_k_labels[topk][tid] = [(topk_label, true_label)]
                                else:
                                    print(top_k_labels[topk][tid])
                                    raise Exception("duplicate tables detected")


                            # Sum hits and update dictionary
                            # hits_at_k[topk] += hits_tensor.sum().item()
                            '''
                            assert true_indices.shape[0] == top_k_indices.shape[0], f"Number of true indices {true_indices.shape[0]} should match number of top_k results{top_k_indices.shape[0]}"
                            hits_tensor = torch.zeros_like(true_indices[:, 0])
                            for i in range(top_k_indices.shape[0]):
                                # use torch's broadcasting 
                                hits_tensor[i] = (true_indices[i] == top_k_indices[i]).any().int()
                            hits_list = hits_tensor.cpu().detach().numpy().tolist()
                            '''
                            # total_hits = hits_tensor.sum().item()
                            # hits_at_k[topk] += hits_tensor.sum().item()

                    elif "sotabfull" in task:
                        # evaluating hit-at-k
                        if task == "sotabfull-re":
                            # Here we are still including all first-[CLS], [CLS] pair, even first-[CLS], first-[CLS]
                            mask = batch["mask"].bool()
                            assert mask.shape[0] == filtered_logits.shape[0], f"Mask shape {mask.shape} should match with filtered_logits shape {filtered_logits.shape}"
                            __shape = batch["label"].shape[0]
                            assert mask.sum() == batch["label"].shape[0], f"Number of True in mask {mask.sum()} should match the number of labels {__shape}"
                            filtered_logits = filtered_logits[mask]

                            '''
                            all_preds = (filtered_logits >= math.log(0.5)
                                         ).int().detach().cpu().numpy()
                            '''
                            all_labels = batch["label"].cpu().detach().numpy()
                            # Ignore the very first CLS token
                            idxes = np.where(all_labels > 0)[0]
                            # print(f'inxes shape {idxes.shape}')
                            # tr_pred_list += all_preds[idxes, :].tolist()
                            # tr_true_list += all_labels[idxes, :].tolist()
                            # Evaluating hit-at-k, using Softmax for now. Sigmoid should also give same order?
                            # probabilities = torch.softmax(torch.tensor(filtered_logits[idxes, :]), dim=1)
                            probabilities = torch.sigmoid(torch.tensor(filtered_logits[idxes, :]))
                            #print(probabil
                            true_indices = torch.where(batch["label"][idxes, :] == 1)[1].reshape(-1, 1)
                            total_instances += true_indices.shape[0]
                            # print(f'sotabfull-re total_instances now {total_instances}')

                            for topk in topk_list:
                                '''
                                probabilities = torch.softmax(torch.tensor(filtered_logits[idxes, :]), dim=1)
                                true_indices = torch.where(batch["label"][idxes, :] == 1)[1].reshape(-1, 1)
                                '''
                                _, top_k_indices = torch.topk(probabilities, topk, dim=1)
                                assert true_indices.shape[0] == top_k_indices.shape[0], f"Number of true indices {true_indices.shape[0]} should match number of top_k results{top_k_indices.shape[0]}"
                                hits_tensor = torch.zeros_like(true_indices[:, 0])
                                for i in range(top_k_indices.shape[0]):
                                    # use torch's broadcasting 
                                    hits_tensor[i] = (true_indices[i] == top_k_indices[i]).any().int()
                                hits_list = hits_tensor.cpu().detach().numpy().tolist()
                                # total_hits = hits_tensor.sum().item()
                                hits_at_k[topk] += hits_tensor.sum().item()
                        elif task == "sotabfull":
                            l2v = pickle.load(open('./sotab27/label_to_vec.pkl', 'rb'))
                            v2l = {str(v.index(1)): k for k, v in l2v.items()}
                            print(v2l)
                            if True:
                                mask = batch["mask"].bool()
                                mask_index = 0
                                assert mask.shape[0] == filtered_logits.shape[0], f"Mask shape {mask.shape} should match with filtered_logits shape {filtered_logits.shape}"
                                print(batch['label'].shape)
                                lab_idx = 0
                                for i, tid in enumerate(batch['table_id']):
                                    num_important_columns = batch['cutoff'][i]
                                    important_column_indices = [] 
                                    important_column_indices_per_t = []
                                    this_tab_counter = 0
                                    while len(important_column_indices) < num_important_columns and mask_index < len(batch['mask']):
                                        if batch['mask'][mask_index]:
                                            important_column_indices.append(mask_index)
                                            important_column_indices_per_t.append(this_tab_counter)
                                        this_tab_counter += 1
                                        mask_index += 1
                                    
                                    print(mask_index)
                                    print(important_column_indices)
                                    print(important_column_indices_per_t)
                                    for j, col_index in enumerate(important_column_indices):
                                        this_col_this_t_idx = important_column_indices_per_t[j]
                                        filtered_logits_col = filtered_logits[col_index]
                                        probabilities = torch.sigmoid(filtered_logits_col).unsqueeze(0)
                                        true_indices = torch.where(batch['label'][lab_idx] == 1)[0].reshape(-1, 1)
                                        total_instances += 1

                                        for topk in topk_list:
                                            _temp, top_k_indices = torch.topk(probabilities, topk, dim=1)
                                            hits_tensor = torch.zeros_like(true_indices[:, 0])

                                            for idx in range(top_k_indices.shape[0]):
                                                hits_tensor[idx] = (true_indices[idx] == top_k_indices[idx]).any().int()

                                            hits_at_k[topk] += hits_tensor.sum().item()
                                            true_indices_list = true_indices.cpu().numpy().tolist()[0]
                                            top_k_list = top_k_indices.cpu().numpy().tolist()[0]
                                            print(top_k_list)
                                            true_label = [v2l[str(vi)] for vi in true_indices_list]
                                            topk_label = [v2l[str(vi)] for vi in top_k_list]
                                            if tid not in top_k_labels[topk]:
                                                # top_k_labels[topk][tid] = [[this_col_this_t_idx, top_k_indices, true_indices]]
                                                top_k_labels[topk][tid] = [[this_col_this_t_idx, topk_label, true_label]]
                                            else:
                                                # top_k_labels[topk][tid].append([this_col_this_t_idx, top_k_indices, true_indices])
                                                top_k_labels[topk][tid].append([this_col_this_t_idx, topk_label, true_label])
                                            # top_k_labels[topk][tid].append({'col': col_index, 'pred': top_k_indices, 'gt': true_indices})
                                            print(top_k_labels[topk][tid])

                                        lab_idx += 1

            with open('topk-turl5652.pkl', 'wb') as f:
                pickle.dump(top_k_labels, f)

            if "sato" in task:
                raise ValueError("sato not supported")
            elif "turl" in task:
                raise ValueError("sato not supported")
            
            print(f'============For task {task} with model {f1_name}============')
            for k, hits in hits_at_k.items():
                rate = hits / total_instances
                print(f'Hits-at-{k} rate is {rate}')
