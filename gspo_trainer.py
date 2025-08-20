# https://www.creetz.com/grpo.html
#
# DDP launch for 8 gpus
# torchrun --standalone --nproc_per_node=8 gspo_train.py
# #
import os
import re
import math
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
import wandb

from utils import (
    extract_hash_answer, 
    extract_xml_answer, 
    compute_format_score, 
    compute_reward, 
    get_lr,
    debug_tensor,
    safe_advantages
)

SYSTEM_PROMPT = (
    """
    A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
    The assistant first thinks about the reasoning process in the mind and then provides the user
    with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and
    <answer> </answer> tags, respectively.
    Example:
    <reasoning> ... </reasoning>
    <answer>42</answer>
    """
)
TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single integer."

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
clip_range = 3e-4  
kl_coef = 0.001 
batch_size = 4     
G = 4
inner_iters = 4    
num_epochs = 1
max_new_tokens = 256
max_grad_norm = 0.2
weight_decay = 0.1  
initial_learning_rate = 3e-6



# setup torch distributed
dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
master_process = rank == 0

seed = 1 + rank
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

policy_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(local_rank)
policy_model.train()
ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(local_rank)
ref_model.eval()

for p in ref_model.parameters():
    p.requires_grad = False

policy_model = DDP(policy_model, device_ids=[local_rank], output_device=local_rank)

param_dict = {pn: p for pn, p in policy_model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr=initial_learning_rate)
optimizer.zero_grad(set_to_none=True)

# setup log
if master_process:
    wandb.init(
        project="gspo-training",
        name=f"gspo-qwen2.5-1.5b-{rank}",
        config={
            "model_name": model_name,
            "clip_range": clip_range,
            "kl_coef": kl_coef,
            "G": G,
            "learning_rate": initial_learning_rate
        }
    )

    log_dir = "./gspo_models"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:
        pass

# setup dataset
train_data = load_from_disk('gsm8k_data/')["train"]

class WrapperDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["question"], item["answer"]

train_data = WrapperDataset(train_data)

def collate_fn(batch):
    prompts, answers = [], []
    for question, answer in batch:
        chat_prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
            {'role': 'user', 'content': "What is 2+2?"},
            {'role': 'assistant', 'content': "<reasoning>To calculate 2+2, we simply add the numbers together: 2 + 2 = 4.</reasoning>\n<answer>4</answer>"},
            {'role': 'user', 'content': question},
        ]
        prompt = tokenizer.apply_chat_template(
            chat_prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
        answers.append(extract_hash_answer(answer))

    return prompts, answers

train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    sampler=train_sampler,
    collate_fn=collate_fn,
    num_workers=4
)

max_steps = len(train_loader)
if master_process:
    print(f"Will run {max_steps} steps per epoch.")

# start training loop
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    for step, (prompts, answers) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        try:
            prompt_enc = tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                padding_side='left',
                truncation=True
            )

            input_ids = prompt_enc["input_ids"].to(local_rank)
            attention_mask = prompt_enc["attention_mask"].to(local_rank)

            # generate G samples per prompt
            policy_model.eval()
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    explore_generations = policy_model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        num_return_sequences=G,
                        top_p=0.9,
                        temperature=1.0,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,  # Add this
                    )
            policy_model.train()
            
            prompt_len = input_ids.shape[1]
            current_batch_size = input_ids.shape[0]

            # compute masks and labels
            batch_attention_mask = (explore_generations != tokenizer.pad_token_id).long()
            batch_action_mask = batch_attention_mask.clone()
            batch_action_mask[:, :prompt_len] = 0
            labels = explore_generations.clone()
            labels[batch_action_mask == 0] = -100

            # compute old logprobs (from policy model)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_old = policy_model(explore_generations, batch_attention_mask, use_cache=False)
                logits_old = out_old.logits
            
            logits_old = logits_old[:, :-1, :].contiguous()
            labels_old = labels[:, 1:].contiguous()
            
            # Check for problematic logits
            if not debug_tensor(logits_old, "logits_old"):
                if master_process:
                    print("Skipping batch due to problematic logits_old")
                continue
            
            logprobs_old = -F.cross_entropy(
                logits_old.view(-1, logits_old.shape[-1]), 
                labels_old.view(-1), 
                reduction='none', 
                ignore_index=-100
            ).view(logits_old.shape[0], -1)

            logprobs_old = logprobs_old.view(current_batch_size, G, -1)

            # compute ref logprobs
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_ref = ref_model(explore_generations, batch_attention_mask, use_cache=False)
                logits_ref = out_ref.logits
            
            logits_ref = logits_ref[:, :-1, :].contiguous()
            labels_ref = labels[:, 1:].contiguous()
            
            if not debug_tensor(logits_ref, "logits_ref"):
                if master_process:
                    print("Skipping batch due to problematic logits_ref")
                continue
            
            logprobs_ref = -F.cross_entropy(
                logits_ref.view(-1, logits_ref.shape[-1]), 
                labels_ref.view(-1), 
                reduction='none', 
                ignore_index=-100
            ).view(logits_ref.shape[0], -1)

            logprobs_ref = logprobs_ref.view(current_batch_size, G, -1)

            # compute advantages
            batch_response_ids = explore_generations[:, prompt_len:]
            batch_response = tokenizer.batch_decode(batch_response_ids, skip_special_tokens=True)
            batch_answers = [extract_xml_answer(batch_response[i]) for i in range(len(batch_response))]
            answers_G = [a for a in answers for _ in range(G)]
            
            batch_format_scores = compute_format_score(batch_response)
            batch_reward_scores = compute_reward(batch_answers, answers_G)
            
            # Use float32 for rewards to avoid precision issues
            batch_rewards = torch.tensor(
                [bfs + brs for bfs, brs in zip(batch_format_scores, batch_reward_scores)], 
                dtype=torch.float32,  # Changed from float16
                device=local_rank
            )
            batch_rewards = batch_rewards.view(current_batch_size, G)

            # Use safe advantage computation
            batch_advantages = safe_advantages(batch_rewards)
            
            # inner iter
            for inner_iter in range(inner_iters):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_new = policy_model(explore_generations, batch_attention_mask, use_cache=False)
                
                logits_new = out_new.logits[:, :-1, :].contiguous()
                labels_new = labels[:, 1:].contiguous()
                
                if not debug_tensor(logits_new, "logits_new"):
                    if master_process:
                        print("Skipping inner iteration due to problematic logits_new")
                    break
                
                logprobs_new = -F.cross_entropy(
                    logits_new.view(-1, logits_new.shape[-1]), 
                    labels_new.view(-1), 
                    reduction='none', 
                    ignore_index=-100
                ).view(logits_new.shape[0], -1)
                
                logprobs_new = logprobs_new.view(current_batch_size, G, -1)

                # gspo loss computation
                valid_mask = batch_action_mask[:, :-1].contiguous().float().view(current_batch_size, G, -1)
                seq_lens = valid_mask.sum(dim=-1)
                empty_sequences = (seq_lens == 0)

                if empty_sequences.any():
                    if master_process:
                        print(f"Warning: {empty_sequences.sum().item()} empty sequences, skipping batch")
                    break

                # sequence level logprobs
                logprobs_old_seq = (logprobs_old * valid_mask).sum(dim=-1)
                logprobs_ref_seq = (logprobs_ref * valid_mask).sum(dim=-1)  
                logprobs_new_seq = (logprobs_new * valid_mask).sum(dim=-1)

                # length normalization
                logprobs_old_seq = logprobs_old_seq / (seq_lens + 1e-8)
                logprobs_ref_seq = logprobs_ref_seq / (seq_lens + 1e-8)
                logprobs_new_seq = logprobs_new_seq / (seq_lens + 1e-8)

                if not debug_tensor(logprobs_old_seq, "logprobs_old_seq") or \
                   not debug_tensor(logprobs_new_seq, "logprobs_new_seq") or \
                   not debug_tensor(logprobs_ref_seq, "logprobs_ref_seq"):
                    if master_process:
                        print("Skipping due to problematic sequence logprobs")
                    break

                log_ratio = torch.clamp(logprobs_new_seq - logprobs_old_seq, min=-10, max=10)
                ratio_seq = torch.exp(log_ratio)
                ratio_seq_clipped = torch.clamp(ratio_seq, 1.0 - clip_range, 1.0 + clip_range)

                if master_process:
                    wandb.log({
                        "mean_reward": batch_rewards.mean().item(),
                        "mean_format_score": torch.tensor(batch_format_scores).mean().item(),
                        "mean_task_score": torch.tensor(batch_reward_scores).mean().item(),
                        "clipped_fraction": (ratio_seq != ratio_seq_clipped).float().mean().item(),
                        "step": step
                    })


                # PPO loss terms
                individual_ppo_reward = torch.min(
                    ratio_seq * batch_advantages,
                    ratio_seq_clipped * batch_advantages
                )

                # KL penalty
                kl_log_ratio = torch.clamp(logprobs_ref_seq - logprobs_new_seq, min=-10, max=10)
                kl_ratio = torch.exp(kl_log_ratio)
                individual_kl_penalty = kl_ratio - kl_log_ratio - 1

                # gspo_loss = -(individual_ppo_reward - kl_coef * individual_kl_penalty).mean()
                # without kl penalty
                gspo_loss = -individual_ppo_reward.mean()

                if torch.isnan(gspo_loss).any():
                    if master_process:
                        print("ERROR: NaN loss detected, skipping batch")
                    break

                # log loss
                loss_value = gspo_loss.detach().clone()
                dist.all_reduce(loss_value, op=dist.ReduceOp.AVG)
                
                if master_process:
                    print(f'gspo training loss at step {step} with ppo epoch {inner_iter} is {loss_value:.6f}')
                    with open(log_file, "a") as f:
                        f.write(f'gspo training loss at step {step} with ppo_epoch {inner_iter} is: {loss_value:.6f}\n')

                gspo_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=max_grad_norm)
                
                lr = get_lr(step, max_steps, max_lr=initial_learning_rate)
                
                if master_process:
                    wandb.log({
                        "train_loss": loss_value.item(),
                        "step": step,
                        "ppo_epoch": inner_iter,
                        "learning_rate": lr
                    })
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # checkpoint
            if master_process and (step % 50 == 0 or step == max_steps - 1):
                ckpt_dir = f"{log_dir}/qwen2.5-1-5b-gspo-step{step+1}"
                policy_model.module.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

        except Exception as e:
            if master_process:
                print(f"Error in step {step}: {e}")
                print("Continuing to next batch...")
            continue

dist.barrier()
dist.destroy_process_group()
