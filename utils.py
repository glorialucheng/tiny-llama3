from dataclasses import dataclass
import numpy as np


@dataclass
class TrainingArgs:
    batch_size: int = 4
    device: str | list = "cuda:0"
    learning_rate: float = 1e-3
    epoch: int = 2
    warming_up_steps: int = 800
    save_path: str = r"./ckpt"
    world_siz: int = -1
    dataset: str = "wiki"

def collate_function(batchs, tokenizer):
    batchs = list(map(tokenizer.encode, batchs))
    max_len = len(max(batchs, key=len))
    for s in batchs:
        s += [tokenizer.pad_id] * (max_len - len(s))
    batchs = np.array(batchs, dtype=int)
    return {"input": batchs[:, :-1], "label": batchs[:, 1:]}