from functools import reduce
import operator
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers
import tqdm


def collate_fn(samples):
    data = torch.nn.utils.rnn.pad_sequence(
        [sample["data"] for sample in samples])
    label = torch.cat([sample["label"] for sample in samples])
    tabs_id = [sample["table_id"] for sample in samples] #added tabs id
    batch = {"data": data, "label": label, "table_id": tabs_id} #added tabs id
    if "idx" in samples[0]:
        # For debug purpose
        batch["idx"] = torch.cat([sample["idx"] for sample in samples])
    return batch

def collate_fn_sotabfull(samples):
    data = torch.nn.utils.rnn.pad_sequence(
        [sample["data"] for sample in samples])
    label = torch.cat([sample["label"] for sample in samples])
    mask = torch.cat([sample["mask"] for sample in samples])
    tabs_id = [sample["table_id"] for sample in samples] #added tabs id
    cutoff = [len(sample["label"]) for sample in samples]
    batch = {"data": data, "label": label, "mask": mask, "table_id": tabs_id, "cutoff": cutoff} #added tabs id
    if "idx" in samples[0]:
        # For debug purpose
        batch["idx"] = torch.cat([sample["idx"] for sample in samples])
    return batch


class SchAugTablewiseDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 folder: str,
                 max_length: int = 32,
                 max_colnum: int = 15, #changed to 10 from 15
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')
            
        if os.path.exists(f"data/{folder}serialization/sa_{split}_tablewise.pkl"):
            print(f"Loading already processed {split} dataset")
            with open(f"data/{folder}serialization/sa_{split}_tablewise.pkl", "rb") as f:
                df_dict = pickle.load(f)
            #Load as dataframe
            self.table_df = df_dict
            
        else:
            try:
                os.mkdir(os.path.join(f"data/{folder}serialization/"))
            except FileExistsError:
                pass
            
            with open(filepath, "rb") as fin:
                df_dict = pickle.load(fin)

            assert split in df_dict
            self.df = df_dict[split]
            self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

            # For learning curve
            num_tables = len(self.df.groupby("table_id"))
            num_train = int(train_ratio * num_tables)

            data_list = []
            for i, (index, group_df) in tqdm.tqdm(enumerate(self.df.groupby("table_id")), total=num_tables):
                if i >= num_train:
                    break

                # It's probably already sorted but just in case.
                group_df = group_df.sort_values("column_index")
                if len(group_df) > max_colnum: #split == "train" and 
                     continue

                token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                    x, add_special_tokens=True, max_length=max_length)).tolist(
                    )
                token_ids = torch.LongTensor(reduce(operator.add,
                                                    token_ids_list)).to(device)
                cls_index_list = [0] + np.cumsum(
                    np.array([len(x) for x in token_ids_list])).tolist()[:-1]
                for cls_index in cls_index_list:
                    assert token_ids[
                        cls_index] == tokenizer.cls_token_id, "cls_indexes validation"


                label_ids_list = group_df["label_ids"].tolist()
                if label_ids_list[0].count(1) == 0 or label_ids_list[0].count(1) > 10:
                    print(f'len of label_id_list[0] is {len(label_ids_list[0])}')
                    print(f'num of classes {label_ids_list[0].count(1)}')
                    continue
                assert label_ids_list[0].count(1) > 0, "In TURL SA, the first column should has all the labels for the table."
                # We do not need mask b/c only the firs col has not-None label_ids
                # mask = [1 if label_id is not None else 0 for label_id in label_ids_list] # mask for columsn where we have labels
                # valid_label_indexes = [cls_index for cls_index, label_id in zip(cls_index_list, label_ids_list) if label_id is not None] # indexes of [CLS] tokens where we have labels.

                # TODO: a list?
                class_ids = torch.LongTensor([label_ids_list[0]]).to(device)
                # cls_indexes = torch.LongTensor(valid_label_indexes).to(device)
                # mask_tensor = torch.BoolTensor(mask).to(device)
                # assert mask_tensor.sum() == len(valid_label_indexes), "mask tensor validation failed"

                data_list.append([index, len(group_df), token_ids, class_ids])

            self.table_df = pd.DataFrame(data_list, columns=[
                                            "table_id", "num_col", "data_tensor", "label_tensor"])

            if multicol_only:
                self.table_df = self.table_df[self.table_df["num_col"] > 1]
                
            with open(f"data/{folder}serialization/sa_{split}_tablewise.pkl", 'wb') as f:
                pickle.dump(self.table_df, f)

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "table_id": self.table_df.iloc[idx]["table_id"]
        }

