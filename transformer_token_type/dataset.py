import os
import torch
from typing import Union, List, Tuple, Dict
from torch.utils.data import Dataset


class TokenTypesDataset(Dataset):
    def __init__(self, folder: str, train: bool = True, vocabs: Tuple[Dict[str, int], Dict[int, str]] = None, max_length: int = None):
        self.all_data: List[List[str]] = []
        
        if train:
            self.idx2token: Dict[int, str] = {}
            self.token2idx: Dict[str, int] = {}
            self.max_length = -1

            for file_name in os.listdir(folder):
                with open(os.path.join(folder, file_name)) as file:
                    for line in file:
                        tokens = line.split()
                        self.max_length = max(self.max_length, len(tokens))
                        for token in tokens:
                            if token not in self.token2idx:
                                self.token2idx[token] = len(self.token2idx) + 4
                                self.idx2token[self.token2idx[token]] = token
                        self.all_data.append(tokens)

            self.max_length = 256
        
        else:
            self.token2idx = vocabs[0]
            self.idx2token = vocabs[1]
            self.max_length = max_length

            for file_name in os.listdir(folder):
                with open(os.path.join(folder, file_name)) as file:
                    for line in file:
                        tokens = line.split()
                        self.all_data.append(tokens)

        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        self.idx2token[self.unk_id] = "<UNK>"

        self.vocab_size = len(self.token2idx) + 4

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        res = []
        for token in tokens:
            if token not in self.token2idx:
                res.append(self.unk_id)
                continue
            res.append(self.token2idx[token])
        return res

    def ids2tokens(self, ids: List[int]) -> List[str]:
        res = []
        for idx in ids:
            if idx == self.pad_id or idx == self.bos_id or idx == self.eos_id:
                continue
            res.append(self.idx2token[idx])
        return res

    def __len__(self):
        """
        Size of the dataset
        :return: number of texts in the dataset
        """
        return len(self.all_data)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: encoded text indices and its actual length (including BOS and EOS specials)
        """

        tokens = [self.bos_id] + self.tokens2ids(self.all_data[item][:self.max_length - 2]) + [self.eos_id]
        padded = torch.full((self.max_length, ), self.pad_id, dtype=torch.int64)
        padded[:len(tokens)] = torch.tensor(tokens)
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Take corresponding index array from self.indices,
        add special tokens (self.bos_id and self.eos_id) and 
        pad to self.max_length using self.pad_id.
        Return padded indices of size (max_length, ) and its actual length
        """
        return padded
