from torch.utils.data import IterableDataset, DataLoader, Dataset
import numpy as np
import os
import json


class WikiDataset(IterableDataset):
    def __init__(self, root, max_seq_len):
        self.files = self._list_files(root)
        self.files.sort()
        self.max_len = max_seq_len
        length = 0
        for file_path in self.files:
            with open(file_path, encoding="utf-8") as f:
                for _ in f:
                    length += 1
        self.length = length

    def __iter__(self):
        return self._sample_generator()

    def __len__(self):
        return self.length

    def _sample_generator(self):
        for file_path in self.files:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    text = json.loads(line)["text"].strip()
                    words = text.split()
                    text = " ".join(words)
                    for i in range(0, len(text), self.max_len):
                        yield text[i : i + self.max_len]

    @staticmethod
    def _list_files(directory, files_list=None):
        if files_list is None:
            files_list = []

        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if os.path.isdir(path):
                WikiDataset._list_files(path, files_list)
            else:
                files_list.append(path)

        return files_list
    

class DistributedIterableDataset(IterableDataset):
    def __init__(self, ite_dataset, rank, world_size):
        self.dataset = ite_dataset
        self.rank = rank
        self.world_size = world_size
        self.length = len(ite_dataset)
        
    def __iter__(self):
        for i, data in enumerate(self.dataset):
            if i % self.world_size == self.rank:
                yield data
    
    def __len__(self):
        return self.length


class OscarDataset(IterableDataset):
    def __init__(self, root, max_seq_len):
        self.files = WikiDataset._list_files(root)
        self.files.sort()
        self.max_seq_len = max_seq_len
        # length = 0
        # for file_path in self.files:
        #     with open(file_path, encoding="utf-8") as f:
        #         for d in f:
        #             d = json.loads(d)
        #             for text in d['段落']:
        #                 for i in range(0, len(text['内容']), max_seq_len):
        #                     length += 1
        # self.length = length
        self.length = 25214361
        # self.length = 24432194
        
    def __iter__(self):
        for file_path in self.files:
            with open(file_path, encoding="utf-8") as f:
                for d in f:
                    d = json.loads(d)
                    for text in d['段落']:
                        for i in range(0, len(text['内容']), self.max_seq_len):
                            yield text['内容'][i:i+self.max_seq_len]

    def __len__(self):
        return self.length