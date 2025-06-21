'''
Disclaimer: This code is modification based on DODUO's codebase to support SA task.
'''

import argparse
import json
import math
import os
import random
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import (
    collate_fn,
    collate_fn_sotabfull,
    SOTABFullColTypeTablewiseDataset,
    SOTABFullRelExtTablewiseDataset,
    SchAugTablewiseDataset,
    TURLColTypeColwiseDataset,
    TURLColTypeTablewiseDataset,
    SatoCVColwiseDataset,
    SatoCVTablewiseDataset,
)

from model import BertForMultiOutputClassification, BertMultiPairPooler
from util import f1_score_multilabel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shortcut_name",
        default="bert-base-uncased",
        type=str,
        help="Huggingface model shortcut name ",
    )
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--epoch",
        default=50,
        # default=30,
        type=int,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--random_seed",
        default=4649,
        type=int,
        help="Random seed",
    )

    parser.add_argument(
        "--num_classes",
        # default=82,
        default=27,
        type=int,
        help="Number of classes",
    )
    parser.add_argument(
        "--num_classes_cpa",
        # default=108,
        default=41,
        type=int,
        help="Number of cpa classes",
    )
    parser.add_argument(
        "--num_classes_sa",
        default=5652,
        type=int,
        help="Number of sa header vocab",
    )
    #TOCHECK
    parser.add_argument("--multi_gpu",
                        action="store_true",
                        default=False,
                        help="Use multiple GPU")
    #TODO
    parser.add_argument("--fp16",
                        action="store_true",
                        default=False,
                        help="Use FP16")
    parser.add_argument("--warmup",
                        type=float,
                        default=0.,
                        help="Warmup ratio")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--tasks",
                        type=str,
                        nargs="+",
                        default=["sa"],
                        help="Task names}")
    parser.add_argument("--colpair",
                        action="store_true",
                        help="Use column pair embedding")
    parser.add_argument("--train_ratios",
                        type=str,
                        nargs="+",
                        default=[],
                        help="e.g., --train_ratios turl=0.8 turl-re=0.1")
    parser.add_argument("--from_scratch",
                        action="store_true",
                        help="Training from scratch")
    parser.add_argument("--single_col",
                        default=False,
                        action="store_true",
                        help="Training with single column model")
    parser.add_argument(
        "--folder",
        default="",
        type=str,
    )

    args = parser.parse_args()
    args.tasks = sorted(args.tasks)
    folder = args.folder

    task_num_class_dict = {
        "sato0": 78,
        "sato1": 78,
        "sato2": 78,
        "sato3": 78,
        "sato4": 78,
        "msato0": 78,
        "msato1": 78,
        "msato2": 78,
        "msato3": 78,
        "msato4": 78,
        "turl": args.num_classes,
        "turl-re": args.num_classes_cpa,
        "sotabfull": args.num_classes,
        "sotabfull-re": args.num_classes_cpa,
        "sa": args.num_classes_sa
    }

    train_ratio_dict = {}
    num_classes_list = []
    for task in args.tasks:
        num_classes_list.append(task_num_class_dict[task])
        # Default training ratio is ALL
        train_ratio_dict[task] = 1.0

    # Training ratio
    for train_ratio in args.train_ratios:
        task, ratio_str = train_ratio.split("=")
        ratio = float(ratio_str)
        assert task in train_ratio_dict, "Invalid task name: {}".format(task)
        assert 0 < ratio <= 1
        train_ratio_dict[task] = ratio

    # For tagname
    train_ratio_str_list = []
    for task in sorted(train_ratio_dict.keys()):
        ratio = train_ratio_dict[task]
        train_ratio_str_list.append("{}-{:.2f}".format(task, ratio))

    if args.colpair:
        assert "turl-re" in args.tasks or "sotabfull-re" in args.tasks, "colpair can be only used for Relation Extraction"

    print("args={}".format(json.dumps(vars(args))), flush=True)

    max_length = args.max_length
    batch_size = args.batch_size
    num_train_epochs = args.epoch

    shortcut_name = args.shortcut_name

    if args.single_col:
        # Single column
        tag_name_col = "single"
    else:
        tag_name_col = "mosato"

    if args.colpair:
        taskname = "{}-colpair".format("".join(args.tasks))
    else:
        taskname = "".join(args.tasks)


    tokenizer = BertTokenizer.from_pretrained(shortcut_name)
    # model = BertForSequenceClassification.from_pretrained(

    models = []
    for i, num_classes in enumerate(num_classes_list):
        print(f'num_classes: {num_classes}')
        if args.single_col:
            model_config = BertConfig.from_pretrained(shortcut_name,
                                                      num_labels=num_classes)
            model = BertForSequenceClassification(model_config)
        else:
            if args.from_scratch:
                # No pre-trained checkpoint
                model_config = BertConfig.from_pretrained(
                    shortcut_name, num_labels=num_classes)
                #TODO: for eval-sotab-doduo
                model = BertForMultiOutputClassification(model_config)
                # model = BertModelMultiOutput(model_config)
            else:
                # Pre-trained checkpoint
                #TODO: for eval-sotab-doduo
                model = BertForMultiOutputClassification.from_pretrained(
                # model = BertModelMultiOutput(model_config).from_pretrained(
                    shortcut_name,
                    num_labels=num_classes,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            if (args.tasks[i] == "turl-re" or args.tasks[i] == "sotabfull-re") and args.colpair:
                print("Use column-pair pooling")
                # Use column pair embeddings
                config = BertConfig.from_pretrained(shortcut_name)
                model.bert.pooler = BertMultiPairPooler(config).to(device)

        # For multi-task learning
        if i > 0:
            assert not args.single_col, "TODO: Single-column model for multi-task learning"
            # The multi-task model shares embeddings & encoder part, not sharing the pooling layer
            model.bert.embeddings = models[0].bert.embeddings
            model.bert.encoder = models[0].bert.encoder
            # [Option] The following also shares the pooling layer
            # model.bert = models[0].bert

        models.append(model.to(device))

    # Check if the parameters are shared
    assert 1 == len(
        set([
            model.bert.embeddings.word_embeddings.weight.data_ptr()
            for model in models
        ]))
    assert 1 == len(
        set([
            model.bert.encoder.layer[0].attention.output.dense.weight.data_ptr(
            ) for model in models
        ]))
    assert len(models) == len(
        set([model.bert.pooler.dense.weight.data_ptr() for model in models]))

    train_datasets = []
    train_dataloaders = []
    valid_datasets = []
    valid_dataloaders = []

    for task in args.tasks:
        train_ratio = train_ratio_dict[task]
        if task in [
                "sato0", "sato1", "sato2", "sato3", "sato4", "msato0",
                "msato1", "msato2", "msato3", "msato4"
        ]:
            pass
        elif task in ['sa']:
            if task in ["turl"]:
                print(task)
                raise ValueError("turl tasks must be turl or turl-re or sa.")
            elif task in ["turl-re"]:
                # TODO: Double-check if it is compatible with single/multi-column data
                filepath = f"data/{folder}table_col_type_serialized.pkl"
                if args.single_col:
                    assert task == "turl"  # Single-column model cannot be used for turl-sch
                    # ColumnWise
                    dataset_cls = TURLColTypeColwiseDataset
                else:
                    # Tablewise
                    dataset_cls = TURLColTypeTablewiseDataset
            elif task in ["sa"]:
                filepath = f"data/{folder}table_sche_aug_serialized_turl_full.pkl"
                dataset_cls = SchAugTablewiseDataset

            else:
                print(task)
                raise ValueError("turl tasks must be turl or turl-re or sa.")

            print('loading sa')
            train_dataset = dataset_cls(filepath=filepath,
                                        split="train",
                                        tokenizer=tokenizer,
                                        folder=folder,
                                        max_length=max_length,
                                        multicol_only=False,
                                        train_ratio=train_ratio,
                                        device=device)
            valid_dataset = dataset_cls(filepath=filepath,
                                        split="dev",
                                        tokenizer=tokenizer,
                                        folder=folder,
                                        max_length=max_length,
                                        multicol_only=False,
                                        device=device)

            # Can be the same
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=batch_size,
                                          collate_fn=collate_fn)
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=batch_size,
                                          collate_fn=collate_fn)
        elif "sotabfull" in task:
            if task in ["sotabfull"]:
                # TODO: Double-check if it is compatible with single/multi-column data
                filepath = f"data/{folder}table_col_type_serialized_sotab_full.pkl"
                '''
                if args.single_col:
                    assert task == "turl"  # Single-column model cannot be used for turl-sch
                    # ColumnWise
                    dataset_cls = TURLColTypeColwiseDataset
                else:
                '''
                # Tablewise
                dataset_cls = SOTABFullColTypeTablewiseDataset
            elif task in ["sotabfull-re"]:
                # TODO: Double-check if it is compatible with single/multi-column data
                #TODO: for eval-sotab-doduo
                filepath = f"data-eval/{folder}table_rel_extraction_serialized_sotab_full.pkl"
                '''
                if args.single_col:
                    assert task == "turl-re"  # Single-column model cannot be used for turl-sch
                    dataset_cls = TURLRelExtColwiseDataset
                else:
                '''
                # Tablewise
                dataset_cls = SOTABFullRelExtTablewiseDataset
            else:
                raise ValueError("sotab tasks must be sotab or sotab-re.")

            print('loading sotab')
            train_dataset = dataset_cls(filepath=filepath,
                                        split="train",
                                        tokenizer=tokenizer,
                                        folder=folder,
                                        max_length=max_length,
                                        multicol_only=False,
                                        train_ratio=train_ratio,
                                        device=device)

            valid_dataset = dataset_cls(filepath=filepath,
                                        split="dev",
                                        tokenizer=tokenizer,
                                        folder=folder,
                                        max_length=max_length,
                                        multicol_only=False,
                                        device=device)

            # Can be the same
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=batch_size,
                                          collate_fn=collate_fn_sotabfull)
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=batch_size,
                                          collate_fn=collate_fn_sotabfull)
        else:
            raise ValueError("task name must be either sato or turl or sotab")

        # Store dataloaders
        train_datasets.append(train_dataset)
        train_dataloaders.append(train_dataloader)
        valid_datasets.append(valid_dataset)
        valid_dataloaders.append(valid_dataloader)

    optimizers = []
    schedulers = []
    loss_fns = []
    for i, train_dataloader in enumerate(train_dataloaders):
        t_total = len(train_dataloader) * num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in models[i].named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0
            },
            {
                "params": [
                    p for n, p in models[i].named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-3, eps=1e-8) 
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=t_total)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

        if "sato" in args.tasks[i]:
            loss_fns.append(CrossEntropyLoss())
        elif "turl" or "sotab" or "sa" in args.tasks[i]:
            loss_fns.append(BCEWithLogitsLoss())
        else:
            raise ValueError("task name must be either sato or turl.")

    set_seed(args.random_seed)

    #TODO: for eval-sotab-doduo
    if args.from_scratch:
        tag_name = "model-doduo-eval/{}/{}/{}_{}_bert_{}-bs{}-ml-{}".format(
            folder, args.random_seed, taskname, tag_name_col, "{}-fromscratch".format(shortcut_name),
            batch_size, max_length)
    else:
        tag_name = "model-doduo-eval/{}/{}/{}_{}_bert_{}-bs{}-ml-{}".format(
            folder, args.random_seed, taskname, tag_name_col, shortcut_name, batch_size, max_length)
    
    # TODO: Check
    tag_name += "__{}".format("_".join(train_ratio_str_list))
    # tag_name = f"{i}/"+ tag_name
    print(tag_name)

    # pdb.set_trace()

    dirpath = os.path.dirname(tag_name)
    if not os.path.exists(dirpath):
        print("{} not exists. Created".format(dirpath))
        os.makedirs(dirpath)

    # Best validation score could be zero
    best_vl_micro_f1s = [-1 for _ in range(len(args.tasks))]
    best_vl_macro_f1s = [-1 for _ in range(len(args.tasks))]
    loss_info_lists = [[] for _ in range(len(args.tasks))]
    for epoch in range(num_train_epochs):
        for k, (task, model, train_dataset, valid_dataset, train_dataloader,
                valid_dataloader, optimizer, scheduler, loss_fn,
                loss_info_list) in enumerate(
                    zip(args.tasks, models, train_datasets, valid_datasets,
                        train_dataloaders, valid_dataloaders, optimizers,
                        schedulers, loss_fns, loss_info_lists)):
            t1 = time()

            model.train()
            tr_loss = 0.
            tr_pred_list = []
            tr_true_list = []

            vl_loss = 0.
            vl_pred_list = []
            vl_true_list = []

            for batch_idx, batch in enumerate(train_dataloader):
                if args.single_col:
                    raise ValueError("Should not be training/using single column based")
                    logits = model(batch["data"].T).logits
                    if "sato" in task:
                        tr_pred_list += logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        tr_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                        loss = loss_fn(logits, batch["label"])
                    elif "turl" in task:
                        # TURL & TURL-REL for the single-col case
                        tr_pred_list += (logits >= math.log(0.5)
                                         ).int().detach().cpu().tolist()
                        tr_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                        loss = loss_fn(logits, batch["label"].float())
                    else:
                        raise ValueError(
                            "Invalid task for single-col: {}".format(task))
                else:
                    # Multi-column model
                    logits, = model(batch["data"].T)  # (row, col) is opposite?

                    # Align the tensor shape when the size is 1
                    if len(logits.shape) == 2:
                        logits = logits.unsqueeze(0)
                    # DEBUG===
                    # print("batch['data'].shape={} data['label'].shape={} batch['idx'].shape={}".format(
                    #    batch["data"].shape, batch["label"].shape, batch["idx"].shape))
                    # ===
                    cls_indexes = torch.nonzero(
                        batch["data"].T == tokenizer.cls_token_id)
                    filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                  logits.shape[2]).to(device)
                    for n in range(cls_indexes.shape[0]):
                        i, j = cls_indexes[n]
                        logit_n = logits[i, j, :]
                        filtered_logits[n] = logit_n
                    if "sato" in task:
                        tr_pred_list += filtered_logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        tr_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                    elif "turl" in task:
                        if task == "turl-re":
                            all_preds = (filtered_logits >= math.log(0.5)
                                         ).int().detach().cpu().numpy()
                            all_labels = batch["label"].cpu().detach().numpy()
                            # Ignore the very first CLS token
                            idxes = np.where(all_labels > 0)[0]
                            tr_pred_list += all_preds[idxes, :].tolist()
                            tr_true_list += all_labels[idxes, :].tolist()
                        elif task == "turl":
                            # Threshold value = 0.5
                            tr_pred_list += (filtered_logits >= math.log(0.5)
                                             ).int().detach().cpu().tolist()
                            tr_true_list += batch["label"].cpu().detach(
                            ).numpy().tolist()
                    elif "sotabfull" in task:
                        if task == "sotabfull-re":
                            # Here we are still including all first-[CLS], [CLS] pair, even first-[CLS], first-[CLS]
                            mask = batch["mask"].bool()
                            assert mask.shape[0] == filtered_logits.shape[0], f"Mask shape {mask.shape} should match with filtered_logits shape {filtered_logits.shape}"
                            __shape = batch["label"].shape[0]
                            assert mask.sum() == batch["label"].shape[0], f"Number of True in mask {mask.sum()} should match the number of labels {__shape}"
                            filtered_logits = filtered_logits[mask]

                            all_preds = (filtered_logits >= math.log(0.5)
                                         ).int().detach().cpu().numpy()
                            all_labels = batch["label"].cpu().detach().numpy()
                            # Ignore the very first CLS token
                            idxes = np.where(all_labels > 0)[0]
                            tr_pred_list += all_preds[idxes, :].tolist()
                            tr_true_list += all_labels[idxes, :].tolist()
                        elif task == "sotabfull":
                            # Now we further filter logits to fetch only those we are intersted in
                            # shape of mask should be equal to the number of [CLS] tokens
                            mask = batch["mask"].bool()
                            assert mask.shape[0] == filtered_logits.shape[0], f"Mask shape {mask.shape} should match with filtered_logits shape {filtered_logits.shape}"
                            __shape = batch["label"].shape[0]
                            assert mask.sum() == batch["label"].shape[0], f"Number of True in mask {mask.sum()} should match the number of labels {__shape}"
                            filtered_logits = filtered_logits[mask]
                            # Threshold value = 0.5
                            tr_pred_list += (filtered_logits >= math.log(0.5)
                                             ).int().detach().cpu().tolist()
                            tr_true_list += batch["label"].cpu().detach(
                            ).numpy().tolist()
                    elif "sa" in task:
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

                        all_preds = (filtered_logits >= math.log(0.5)
                                     ).int().detach().cpu().numpy()
                        all_labels = batch["label"].cpu().detach().numpy()
                        # Set threshold to 0.5 manually
                        tr_pred_list += (filtered_logits >= math.log(0.5)
                                         ).int().detach().cpu().tolist()

                        tr_true_list += batch["label"].cpu().detach(
                        ).numpy().tolist()

                        assert len(tr_pred_list) == len(tr_true_list), f"len of tr_pred_list: {len(tr_pred_list)} should match len of tr_true_list: {len(tr_true_list)}"
                        assert tr_true_list[-1].count(1) > 0 and tr_true_list[-1].count(1) < 20, f"Last element of batch true sa has too many headers : {tr_true_list[-1].count(1)}"


                    if "sato" in task:
                        loss = loss_fn(filtered_logits, batch["label"])
                    elif "turl" or "sotabfull" or "sa" in task:
                        loss = loss_fn(filtered_logits, batch["label"].float())


                loss.backward()
                tr_loss += loss.item()
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            tr_loss /= (len(train_dataset) / batch_size)

            if "sato" in task:
                tr_micro_f1 = f1_score(tr_true_list,
                                       tr_pred_list,
                                       average="micro")
                tr_macro_f1 = f1_score(tr_true_list,
                                       tr_pred_list,
                                       average="macro")
                tr_class_f1 = f1_score(tr_true_list,
                                       tr_pred_list,
                                       average=None,
                                       labels=np.arange(args.num_classes))
            elif "turl" or "sotabfull" or "sa" in task:
                tr_micro_f1, tr_macro_f1, tr_class_f1, _ = f1_score_multilabel(
                    tr_true_list, tr_pred_list)

            # Validation
            model.eval()
            for batch_idx, batch in enumerate(valid_dataloader):
                if args.single_col:
                    raise ValueError("Should not be using single-column for valid")
                    # Single-column
                    logits = model(batch["data"].T).logits
                    if "sato" in task:
                        vl_pred_list += logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        vl_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                        loss = loss_fn(logits, batch["label"])
                    elif "turl" in task:
                        tr_pred_list += (logits >= math.log(0.5)
                                         ).int().detach().cpu().tolist()
                        tr_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                        loss = loss_fn(logits, batch["label"].float())
                    else:
                        raise ValueError(
                            "Invalid task for single-col: {}".format(task))
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
                        vl_pred_list += filtered_logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        vl_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                    elif "turl" in task:
                        if task == "turl-re":
                            all_preds = (filtered_logits >= math.log(0.5)
                                         ).int().detach().cpu().numpy()
                            all_labels = batch["label"].cpu().detach().numpy()
                            idxes = np.where(all_labels > 0)[0]
                            vl_pred_list += all_preds[idxes, :].tolist()
                            vl_true_list += all_labels[idxes, :].tolist()
                        elif task == "turl":
                            # Threshold value = 0.5
                            vl_pred_list += (filtered_logits >= math.log(0.5)
                                             ).int().detach().cpu().tolist()
                            vl_true_list += batch["label"].cpu().detach(
                            ).numpy().tolist()
                    elif "sotabfull" in task:
                        if task == "sotabfull-re":
                            # Here we are still including all first-[CLS], [CLS] pair, even first-[CLS], first-[CLS]
                            mask = batch["mask"].bool()
                            assert mask.shape[0] == filtered_logits.shape[0], f"Mask shape {mask.shape} should match with filtered_logits shape {filtered_logits.shape}"
                            __shape = batch["label"].shape[0]
                            assert mask.sum() == batch["label"].shape[0], f"Number of True in mask {mask.sum()} should match the number of labels {__shape}"
                            filtered_logits = filtered_logits[mask]

                            all_preds = (filtered_logits >= math.log(0.5)
                                         ).int().detach().cpu().numpy()
                            all_labels = batch["label"].cpu().detach().numpy()
                            # Ignore the very first CLS token
                            idxes = np.where(all_labels > 0)[0]
                            vl_pred_list += all_preds[idxes, :].tolist()
                            vl_true_list += all_labels[idxes, :].tolist()
                        elif task == "sotabfull":
                            # Now we further filter logits to fetch only those we are intersted in
                            # shape of mask should be equal to the number of [CLS] tokens
                            mask = batch["mask"].bool()
                            assert mask.shape[0] == filtered_logits.shape[0], f"Mask shape {mask.shape} should match with filtered_logits shape {filtered_logits.shape}"
                            __shape = batch["label"].shape[0]
                            assert mask.sum() == batch["label"].shape[0], f"Number of True in mask {mask.sum()} should match the number of labels {__shape}"
                            filtered_logits = filtered_logits[mask]
                            # Threshold value = 0.5
                            vl_pred_list += (filtered_logits >= math.log(0.5)
                                             ).int().detach().cpu().tolist()
                            vl_true_list += batch["label"].cpu().detach(
                            ).numpy().tolist()
                    elif "sa" in task:
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

                        all_preds = (filtered_logits >= math.log(0.5)
                                     ).int().detach().cpu().numpy()
                        all_labels = batch["label"].cpu().detach().numpy()
                        # Set threshold to 0.5 manually
                        vl_pred_list += (filtered_logits >= math.log(0.5)
                                         ).int().detach().cpu().tolist()

                        vl_true_list += batch["label"].cpu().detach(
                        ).numpy().tolist()

                        assert len(vl_pred_list) == len(vl_true_list), f"len of vl_pred_list: {len(vl_pred_list)} should match len of vl_true_list: {len(vl_true_list)}"
                        assert vl_true_list[-1].count(1) > 0 and vl_true_list[-1].count(1) < 20, f"Last element of batch true sa has too many headers"



                    if "sato" in task:
                        loss = loss_fn(filtered_logits, batch["label"])
                    elif "turl" or 'sotabfull' or 'sa' in task:
                        loss = loss_fn(filtered_logits, batch["label"].float())

                vl_loss += loss.item()

            vl_loss /= (len(valid_dataset) / batch_size)
            
            if "sato" in task:
                vl_micro_f1 = f1_score(vl_true_list,
                                       vl_pred_list,
                                       average="micro")
                vl_macro_f1 = f1_score(vl_true_list,
                                       vl_pred_list,
                                       average="macro")
                vl_class_f1 = f1_score(vl_true_list,
                                       vl_pred_list,
                                       average=None,
                                       labels=np.arange(args.num_classes))
            elif "turl" or "sotabfull" or "sa" in task:
                vl_micro_f1, vl_macro_f1, vl_class_f1, _ = f1_score_multilabel(
                    vl_true_list, vl_pred_list)

            if vl_micro_f1 > best_vl_micro_f1s[k]:
                best_vl_micro_f1s[k] = vl_micro_f1
                if len(args.tasks) >= 2:
                    model_savepath = "{}={}_best_micro_f1.pt".format(
                        tag_name, task)
                else:
                    model_savepath = "{}_best_micro_f1.pt".format(tag_name)
                torch.save(model.state_dict(), model_savepath)

            if vl_macro_f1 > best_vl_macro_f1s[k]:
                best_vl_macro_f1s[k] = vl_macro_f1
                if len(args.tasks) >= 2:
                    model_savepath = "{}={}_best_macro_f1.pt".format(
                        tag_name, task)
                else:
                    model_savepath = "{}_best_macro_f1.pt".format(tag_name)
                torch.save(model.state_dict(), model_savepath)

            loss_info_list.append([
                tr_loss, tr_macro_f1, tr_micro_f1, vl_loss, vl_macro_f1,
                vl_micro_f1
            ])
            t2 = time()
            print(
                "Epoch {} ({}): tr_loss={:.7f} tr_macro_f1={:.4f} tr_micro_f1={:.4f} "
                .format(epoch, task, tr_loss, tr_macro_f1, tr_micro_f1),
                "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f} ({:.2f} sec.)"
                .format(vl_loss, vl_macro_f1, vl_micro_f1, (t2 - t1)), flush=True)

    for task, loss_info_list in zip(args.tasks, loss_info_lists):
        loss_info_df = pd.DataFrame(loss_info_list,
                                    columns=[
                                        "tr_loss", "tr_f1_macro_f1",
                                        "tr_f1_micro_f1", "vl_loss",
                                        "vl_f1_macro_f1", "vl_f1_micro_f1"
                                    ])
        if len(args.tasks) >= 2:
            loss_info_df.to_csv("{}={}_loss_info.csv".format(tag_name, task))
        else:
            loss_info_df.to_csv("{}_loss_info.csv".format(tag_name))