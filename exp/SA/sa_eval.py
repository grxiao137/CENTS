import os
import json
import random
import math
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from collections import Counter

from data import sa_ds_map

def load_json_shard(path, idlist=None):
    tids = []
    all_correct_tids = []
    gt, pred = [], []
    res = json.loads(open(path, 'r').read())
    for tid, gt_list in res['gt'].items():
        if idlist and tid not in idlist:
            continue
        _f = False
        for g, p in zip(gt_list, res['pred'][tid]):
            if g.lower() != p.lower():
                _f = True
        if not _f:
            all_correct_tids.append(tid)
        gt.append(gt_list)
        tids.append(tid)
        pred.append(res['pred'][tid])
    return gt, pred, tids, all_correct_tids


def _search_sim(l1, l2, samap):
    f1, f2 = False, False
    if l1 in samap:
        f1 = True
    if l2 in samap:
        f2 = True
    for c in samap:
        if c in l1:
            f1 = True
        if c in l2:
            f2 = True
    return (f1 and f2)


def load_map(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    original_phrase, representative_phrase = parts
                    mapping[original_phrase] = representative_phrase
                else:
                    print(f"Warning: Line format incorrect: '{line.strip()}'")
    return mapping

from sklearn.metrics import average_precision_score
import numpy as np

"""Information Retrieval metrics
Useful Resources:
http://www.cs.utexas.edu/~mooney/ir-course/slides/Evaluation.ppt
http://www.nii.ac.jp/TechReports/05-014E.pdf
http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
http://hal.archives-ouvertes.fr/docs/00/72/67/60/PDF/07-busa-fekete.pdf
Learning to Rank for Information Retrieval (Tie-Yan Liu)
"""
import numpy as np
import pdb
import os
import pickle


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

import numpy as np

def mean_recall(relevance_lists, total_relevant_docs):
    """
    Calculates the mean recall over multiple instances.

    Args:
        relevance_lists (list of lists): Each inner list contains binary relevance judgments for an instance.
        total_relevant_docs (list): Total number of relevant documents for each instance.

    Returns:
        float: Mean recall over all instances.
    """
    recalls = []
    for r_list, total_rel in zip(relevance_lists, total_relevant_docs):
        r_array = np.array(r_list)
        relevant_retrieved = np.sum(r_array)
        if total_rel == 0:
            recall = 0.0  # Avoid division by zero
        else:
            recall = relevant_retrieved / total_rel
        recalls.append(recall)
    mean_recall = np.mean(recalls)
    return mean_recall




def mean_average_precision_multilabel(true_labels_list, pred_labels_list, tids, path):
    true_labels_list = [ [label.lower() for label in labels] for labels in true_labels_list ]
    pred_labels_list = [ [label.lower() for label in labels] for labels in pred_labels_list ]

    lm = load_map(path)
    in_task_data_ct = []

    unique_classes = set(list(lm.values()))

    filtered_pred_labels_list = []

    for pred_labels in pred_labels_list:
        filtered_labels = []
        for label in pred_labels:
            if label == '':
                filtered_labels.append('OOV')
            elif label not in unique_classes:
                filtered_labels.append('OOV')
            else:
                filtered_labels.append(label)
        filtered_pred_labels_list.append(filtered_labels)

    valid_pred_labels_list = []
    valid_true_labels_list = []
    for i in range(len(true_labels_list)):
        true_labels = true_labels_list[i]
        original_pred_labels = pred_labels_list[i]
        pred_labels = filtered_pred_labels_list[i]

        valid_pred_labels = []
        for pred_label, original_pred_label in zip(pred_labels, original_pred_labels):
            if pred_label != 'OOV':
                valid_pred_labels.append(pred_label)
                continue
            done = False
            for subpred in original_pred_label.split():
                for tl in true_labels:
                    if subpred in tl.split():
                        valid_pred_labels.append(tl)
                        done = True
                        break
                if done:
                    break
            else:
                valid_pred_labels.append(pred_label)

        valid_pred_labels_list.append(valid_pred_labels)
        valid_true_labels_list.append(true_labels)


    res_all = []

    assert len(tids) == len(valid_true_labels_list) == len(valid_pred_labels_list) == len(pred_labels_list), '''Length has to be the same for all'''
    
    for gt_list, pred_list, orig_pred_list, tid in zip(valid_true_labels_list, valid_pred_labels_list, pred_labels_list, tids):
        this_binary = [0] * len(pred_list)
        for i, pred in enumerate(pred_list):
            if i > len(gt_list):
                break
            if pred in gt_list:
                this_binary[i] = 1
            else:
                for samap in sa_ds_map:
                    for gt in gt_list:
                        if _search_sim(orig_pred_list[i], gt, samap):
                            this_binary[i] = 1
        this_binary = this_binary[:len(gt_list)]
        res_all.append(this_binary)

    mAP = mean_average_precision(res_all)

    return mAP

def run(args):
    res_dir = args.dir
    tids = []
    gt, pred = [], []

    for file in os.listdir(res_dir):
        if file.endswith('.json') or file.endswith('.jsonl'):
            _gt, _pred, _tids, _ = load_json_shard(os.path.join(res_dir, file))
            gt += _gt
            pred += _pred
            tids += _tids

    mean_ap  = mean_average_precision_multilabel(gt, pred, tids, args.map)

    print(f'Mean AP: {mean_ap:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SA evaluation')
    parser.add_argument('--dir', type=str, help='Dir to load the output')
    parser.add_argument('--map', type=str)
    args = parser.parse_args()
    run(args)
