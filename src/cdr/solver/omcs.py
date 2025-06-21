import time
import copy
import mosek
import random
import numpy as np
import multiprocessing as mp
from typing import AnyStr, Dict
from .base_solver import Solver

def streamprinter(msg):
    print(msg.strip())

def noprinter(msg):
    pass

class OMCS(Solver):
    def __init__(self):
        pass

    def _pre_process_cs_info(self, alpha, beta, cs2idx, cs2s, cs2w):
        alpha_o = float('-inf')
        alpha_p = float('inf')
        beta_k  = float('-inf')
        beta_l  = float('inf')
        alpha_size = len(cs2idx[alpha])
        beta_size  = len(cs2idx[beta])
        flag_val = -1

        sorted_cs_vals = sorted(cs2idx.keys())
        for cs_val in sorted_cs_vals:
            if cs_val < alpha and cs_val > alpha_o:
                alpha_o = cs_val
            if alpha < cs_val < beta and cs_val < alpha_p:
                alpha_p = cs_val
            if cs_val < beta and cs_val > beta_k:
                beta_k = cs_val
            if cs_val > beta and cs_val < beta_l:
                beta_l = cs_val

        if alpha_o == float('-inf'):
            alpha_o = 0.0
        if alpha_p == float('inf'):
            alpha_p = beta
        if beta_k == float('-inf'):
            beta_k = alpha
        if beta_l == float('inf'):
            beta_l = 1.0

        tav = cs2s[alpha] * alpha 
        tbv = cs2s[beta]  * beta  
        taw = cs2w[alpha] * alpha 
        tbw = cs2w[beta]  * beta         

        flag_val = taw - (tav * tbw) / tbv

        return alpha_o, alpha_p, beta_k, beta_l, alpha_size, beta_size, flag_val


    def _move_two_sets(self, alpha, beta, alpha_o, alpha_p, beta_k, beta_l, alpha_size, beta_size, cs2s, cs2w, flag_val):
        tav, tbv, taw, tbw = cs2s[alpha] * alpha, cs2s[beta] * beta, cs2w[alpha] * alpha, cs2w[beta] * beta
        new_alpha, new_beta = -1, -1
        if flag_val >= 0:
            if alpha - alpha_o < (tbv / tav) * (beta_l - beta):
                new_alpha = alpha_o
                new_beta = beta + (tav / tbv) * (alpha - alpha_o)
            else:
                new_alpha = alpha - (tbv / tav) * (beta_l - beta)
                new_beta = beta_l
        else:
            _temp = min(alpha_p - alpha, (tbv / tav) * (beta - beta_k), (tbv / (tav + tbv)) * (beta - alpha))
            if alpha_p - alpha == _temp:
                new_alpha = alpha_p
                new_beta = beta - (tav / tbv) * (alpha_p - alpha)
            elif (tbv / tav) * (beta - beta_k) == _temp:
                new_alpha = alpha + (tbv / tav) * (beta - beta_k)
                new_beta = beta_k
            else:
                new_alpha = (alpha * tav + beta * tbv) / (tav + tbv)
                new_beta = (alpha * tav + beta * tbv) / (tav + tbv)

        return new_alpha, new_beta

    def _gen_cell_sets(self, y, scores, weights):
        cs2idx = {}
        cs2s, cs2w = {}, {}
        for i, y_i in enumerate(y):
            if self.verbose:
                print(y_i)
            if y_i not in cs2idx:
                cs2idx[y_i] = []
                cs2s[y_i] = 0
                cs2w[y_i] = 0
            cs2idx[y_i].append(i)
            cs2s[y_i] += scores[i]
            cs2w[y_i] += weights[i]
        if 0 not in cs2idx:
            cs2idx[0] = []
            cs2s[0] = 0
            cs2w[0] = 0
        if 1 not in cs2idx:
            cs2idx[1] = []
            cs2s[1] = 0
            cs2w[1] = 0
        return cs2idx, cs2s, cs2w

    def _update_cell_sets(self, alpha, beta, new_alpha, new_beta, cs2idx, cs2s, cs2w):
        if new_alpha not in cs2idx:
            cs2idx[new_alpha] = []
            cs2s[new_alpha] = 0
            cs2w[new_alpha] = 0
        if new_beta not in cs2idx:
            cs2idx[new_beta] = []
            cs2s[new_beta] = 0
            cs2w[new_beta] = 0
        if alpha != new_alpha:
            cs2s[new_alpha] += cs2s[alpha]
            cs2w[new_alpha] += cs2w[alpha]
            for i in cs2idx[alpha]:
                cs2idx[new_alpha].append(i)

            del cs2idx[alpha]
            del cs2s[alpha]
            del cs2w[alpha]
        if beta != new_beta:
            cs2s[new_beta] += cs2s[beta]
            cs2w[new_beta] += cs2w[beta]
            for i in cs2idx[beta]:
                cs2idx[new_beta].append(i)

            del cs2idx[beta]
            del cs2s[beta]
            del cs2w[beta]

        return cs2idx, cs2s, cs2w
    
    def _find_two_cell_sets(self, cs2idx):
        cs_keys = list(cs2idx.keys())
        val1, val2 = 0, 0
        non_zero_one = [v for v in cs_keys if v != 0 and v != 1]
        val1, val2 = random.sample(non_zero_one, 2)
        alpha, beta = sorted([val1, val2])
        return alpha, beta
    
    def _round_column(self, y, scores, weights):
        new_score = 0
        new_weights = 0

        new_y = [-1] * len(y)

        y_copy = copy.deepcopy(y)
        unique_y = np.unique(y_copy)
        idx = (unique_y >= 1e-7) & (unique_y <= 1-1e-7)
        if np.sum(idx) >= 2:
            cs2idx, cs2s, cs2w = self._gen_cell_sets(y_copy, scores, weights)
            while len(list(cs2idx.keys())) > 3:
                alpha, beta = self._find_two_cell_sets(cs2idx)
                alpha_o, alpha_p, beta_k, beta_l, alpha_size, beta_size, flag_val = self._pre_process_cs_info(alpha, beta, cs2idx, cs2s, cs2w)
                if not flag_val:
                    break
                new_alpha, new_beta = self._move_two_sets(alpha, beta, alpha_o, alpha_p, beta_k, beta_l, alpha_size, beta_size, cs2s, cs2w, flag_val)
                cs2idx, cs2s, cs2w = self._update_cell_sets(alpha, beta, new_alpha, new_beta, cs2idx, cs2s, cs2w)
            assert len(list(cs2idx.keys())) == 3, "Should only have 3 unique values as of now"
            for cs_val in cs2idx:
                if cs_val < 1e-3:
                    for idx in cs2idx[cs_val]:
                        new_y[idx] = 0
                elif cs_val > 1-1e-3:
                    for idx in cs2idx[cs_val]:
                        new_y[idx] = 1
                        new_score += scores[idx]
                        new_weights += weights[idx]
                else:
                    for idx in cs2idx[cs_val]:
                        new_y[idx] = 0
        else:
            new_score = 0
            new_y = []
            for val, s in zip(y, scores):
                if val < 1e-3:
                    new_y.append(0)
                elif val > 1-1e-3:
                    new_y.append(1)
                    new_score += s
                

        return new_y, new_score


    def _solve_LP(self, c2c: Dict, c2w: Dict, c2s: Dict, c2b: Dict, total_budget, verbose=False, *kwargs):
        ## check all input data size match, each column should have the same length
        assert len(c2c) == len(c2w) == len(c2s) == len(c2b), "Input column size mismatch"

        for col in c2c:
            assert len(c2c[col]) == len(c2w[col]) == len(c2s[col]), "Input row size mismatch"
        
        ts, tt = 0, 0
        c2sel = {col: [] for col in c2c}

        st = time.time()
        with mosek.Env() as env:
            _ = 0.0
            with env.Task(0, 0) as task:
                if verbose:
                    task.set_Stream(mosek.streamtype.log, streamprinter)
                else:
                    task.set_Stream(mosek.streamtype.log, noprinter)

                num_vars = sum([len(c2c[col]) for col in c2c])
                task.appendvars(num_vars)
                c2idx = {col: [] for col in c2c}
                this_idx = 0
                for col in c2c:
                    base_scores = c2s[col]
                    weights = c2w[col]
                    c2idx[col] = list(range(this_idx, this_idx + len(c2c[col])))
                    for i, score in enumerate(base_scores):
                        ci = this_idx + i
                        task.putcj(ci, score)
                        task.putvarbound(ci, mosek.boundkey.ra, 0.0, 1.0)
                    this_idx += len(c2c[col])

                con_idx = 0
                total_budget = 0
                for col in c2c:
                    col_var_idx, col_var_weights, col_budget = c2idx[col], c2w[col], c2b[col]
                    task.appendcons(1)
                    total_budget += col_budget
                    task.putconbound(con_idx, mosek.boundkey.up, _, col_budget)
                    task.putarow(con_idx, col_var_idx, col_var_weights)
                    con_idx += 1
                
                task.putobjsense(mosek.objsense.maximize)
                task.optimize()
                task.solutionsummary(mosek.streamtype.msg)
                soln = [0.] * num_vars
                task.getxx(mosek.soltype.bas, soln)

                y_values = soln[:num_vars]
                c2sel = {col: [] for col in c2c}
                for col, col_var_id in c2idx.items():
                    c2sel[col] = [y_values[i] for i in col_var_id]


                tt = time.time() - st

                return c2sel, ts, tt

    def solve(self, c2c: Dict, c2w: Dict, c2s: Dict, c2b: Dict, total_budget, verbose=False, *kwargs):
        self.verbose = verbose
        def _assert_keys_aligned(*dicts):
            keys = [d.keys() for d in dicts]
            for key_set in keys[1:]:
                assert key_set == keys[0], "Dictionaries do not have matching keys."
        def _assert_list_lengths(*dicts):
            for key in dicts[0].keys():
                lengths = [len(d[key]) for d in dicts]
                for length in lengths[1:]:
                    assert length == lengths[0], f"Lists for column {key} do not have the same length."

        st = time.time()
        lp_c2sel, lp_ts, lp_tt = self._solve_LP(c2c, c2w, c2s, c2b, total_budget, verbose)
        _assert_keys_aligned(lp_c2sel, c2s, c2w)
        _assert_list_lengths(lp_c2sel, c2s, c2w)
        problems = [(lp_c2sel[col], c2s[col], c2w[col]) for col in lp_c2sel.keys()]

        c2sel={}
        ns = 0
        for p, col in zip(problems, lp_c2sel.keys()):
            cells = c2c[col]
            _ys, _ns = self._round_column(p[0], p[1], p[2])
            c2sel[col] = [cells[idx] for idx, val in enumerate(_ys) if val == 1 or val == 1.0]
            ns += _ns
        if verbose:
            print('='*50)

        et = time.time() - st 

        return c2sel, ns, lp_tt + et

    def __str__(self):
        return f"OMCSSolver()"