from datetime import datetime, timedelta, timezone
from functools import partial
import torch.nn as nn
import torch
from torch import optim
from llama.model import Transformer, ModelArgs
from torch.utils.data import IterableDataset, DataLoader, Dataset
import os
import json
from transformers import get_cosine_schedule_with_warmup
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from llama.tokenizer import Tokenizer
from utils import TrainingArgs, collate_function
import matplotlib.pyplot as plt
import time
from mydataset import WikiDataset, DistributedIterableDataset, OscarDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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
    batch_size=10, device='cuda', learning_rate=1e-5, epoch=2, warming_up_steps=800, dataset='oscar'
)

tokenizer = Tokenizer("./tokenizer/tokenizer.model")
tokenizer.pad_id = 0

def ddp(rank, world_size):
    if rank != 0:
        time.sleep(1)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    # we need to set seed so that models are initialized in the same way
    torch.manual_seed(0)
    model = Transformer(model_args).to(rank)
    model = DDP(model, device_ids=[rank])
    opt = optim.AdamW(params=model.parameters(), lr=training_args.learning_rate)
    criteria = nn.functional.cross_entropy
    if training_args.dataset == 'wiki':
        dataset = WikiDataset(r"../lanyun-tmp/wiki_zh", max_seq_len=model_args.max_seq_len // 2)
    elif training_args.dataset == 'oscar':
        dataset = OscarDataset(r"../lanyun-tmp/oscar", max_seq_len=model_args.max_seq_len // 2)
    dataset = DistributedIterableDataset(dataset, rank, world_size)
    
    # 每个device的batch size
    trainloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=training_args.batch_size, 
                                              num_workers=world_size,
                                              collate_fn=partial(collate_function, tokenizer=tokenizer),
                                              )
                                            #   sampler=sampler)
    cosine_schedule = get_cosine_schedule_with_warmup(
        opt, training_args.warming_up_steps, len(dataset) * training_args.epoch
    )
    
    loss_history = []
    global_step = 0
    for epoch in range(training_args.epoch):
        for i, batch in enumerate(trainloader):
            global_step += 1
            opt.zero_grad()
            input = torch.Tensor(batch["input"]).to(int).to(rank)
            label = torch.Tensor(batch["label"]).to(int).to(rank)
            predict = model(input)
            loss = criteria(
                predict.transpose(1, 2), label, ignore_index=tokenizer.pad_id
            )
            
            # 打印 cuda:0 的信息
            if rank == 0:
                print("=" * 40)
                print("LR:", opt.param_groups[0]["lr"])
                loss_history.append(loss.cpu().detach().numpy().tolist())
                print(
                    f"Epoch: {epoch+1}/{training_args.epoch}, batch: {i+1}/{len(trainloader)}, loss: {loss_history[-1]}"
                )
                print("After step:", torch.cuda.memory_allocated("cuda:0"))
                print("=" * 40)

                if global_step % 40000 == 0:
                    torch.save(
                        model.state_dict(),
                        training_args.save_path
                        + f"/llama-{datetime.now().strftime('%Y%m%d%H%M%S')+'S'+str(global_step)}.pth",
                    )
            loss.backward()
            opt.step()
            cosine_schedule.step()
            
            # 保存 cuda:0 的权重即可，不需要每个 device 都保存
            if rank == 0:
                if global_step % 40000 == 0:
                    torch.save(
                        model.module.state_dict(),
                        training_args.save_path
                        + f"/llama-{training_args.dataset}-{datetime.now().strftime('%Y%m%d%H%M%S')+'S'+str(global_step)}.pth",
                    )
    if rank == 0:
        torch.save(
            model.module.state_dict(),
            training_args.save_path
            + f"/llama-{training_args.dataset}-{datetime.now(tz=timezone(timedelta(hours=8))).strftime('%Y%m%d%H%M%S')+'S'+str(global_step)}.pth",
        )
        plt.plot(loss_history)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        t = datetime.now(tz=timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
        plt.title(t)
        plt.savefig(f'./result/loss-{t}.png')


def main():
    world_size = torch.cuda.device_count()
    mp.spawn(ddp,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()