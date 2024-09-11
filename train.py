from dataclasses import dataclass
from datetime import datetime
from llama import Transformer, ModelArgs, Tokenizer
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import os
import json
from torch.optim import lr_scheduler
from transformers import get_cosine_schedule_with_warmup
from functools import partial
import matplotlib.pyplot as plt
from mydataset import WikiDataset
from utils import TrainingArgs, collate_function


def train():
    model_args = ModelArgs(
        n_layers=2,
        vocab_size=128256,
        dim=1024,
        n_heads=8,
        n_kv_heads=8,
        multiple_of=32,
        max_seq_len=1024,
        max_batch_size=32,
    )

    training_args = TrainingArgs(
        batch_size=8, device="cuda:0", learning_rate=1e-4, epoch=2, warming_up_steps=800
    )

    tokenizer = Tokenizer("./tokenizer/tokenizer.model")
    tokenizer.pad_id = 0
    model = Transformer(model_args).to("cuda")
    opt = optim.AdamW(params=model.parameters(), lr=training_args.learning_rate)
    dataset = WikiDataset(r"../lanyun-tmp/wiki_zh", max_seq_len=model_args.max_seq_len // 2)
    criteria = nn.functional.cross_entropy
    cosine_schedule = get_cosine_schedule_with_warmup(
        opt, training_args.warming_up_steps, len(dataset) * training_args.epoch
    )

    loss_history = []
    global_step = 0
    for epoch in range(training_args.epoch):
        dataloader = DataLoader(
            dataset,
            batch_size=training_args.batch_size,
            collate_fn=partial(collate_function, tokenizer=tokenizer),
        )
        for i, batch in enumerate(dataloader):
            global_step += 1
            opt.zero_grad()
            input = torch.Tensor(batch["input"]).to(int).to(training_args.device)
            label = torch.Tensor(batch["label"]).to(int).to(training_args.device)
            predict = model(input)
            loss = criteria(
                predict.transpose(1, 2), label, ignore_index=tokenizer.pad_id
            )
            print("=" * 40)
            print("LR:", opt.param_groups[0]["lr"])
            loss.backward()
            opt.step()
            cosine_schedule.step()
            loss_history.append(loss.cpu().detach().numpy().tolist())
            print(
                f"Epoch: {epoch+1}/{training_args.epoch}, batch: {i+1}/{len(dataloader)}, loss: {loss_history[-1]}"
            )

            print("After step:", torch.cuda.memory_allocated("cuda:0"))
            print("=" * 40)

            if global_step % 40000 == 0:
                torch.save(
                    model.state_dict(),
                    training_args.save_path
                    + f"/llama-{datetime.now().strftime('%Y%m%d%H%M%S')+'S'+str(global_step)}.pth",
                )
    torch.save(
        model.state_dict(),
        training_args.save_path
        + f"/llama-{datetime.now().strftime('%Y%m%d%H%M%S')+'S'+str(global_step)}.pth",
    )
    return loss_history


if __name__ == "__main__":
    loss = train()
    plt.plot(loss)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.title(t)
    plt.savefig(f'./result/loss-{t}.png')