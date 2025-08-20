import re
import math

def extract_xml_answer(text: str) -> str:
    """Extracts the answer from a response using the <answer> tag."""
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        return ""

def extract_hash_answer(text: str) -> str | None:
    """Extracts the answer from the dataset if it uses the '####' delimiter."""
    try:
        return text.split("####")[1].strip()
    except IndexError:
        return None

def compute_format_score(batch_responses):
    """Reward function that checks if the completion has the correct format."""
    pattern = r"^<reasoning>(?:(?!</reasoning>).)*</reasoning>\n<answer>(?:(?!</answer>).)*</answer>$"
    matches = [bool(re.match(pattern, g_a)) for g_a in batch_responses]
    format_scores = [1.0 if match else 0.0 for match in matches]
    return format_scores

def compute_reward(batch_answers, answers):
    """Reward function that checks if the answer is correct."""
    reward_scores = [2.0 if g_a == a else 0.0 for g_a, a in zip(batch_answers, answers)]
    return reward_scores
    
# learning rate schedule
def get_lr(it, max_steps, warmup_steps = None, max_lr=1e-5, min_lr=1e-6):
    warmup_steps = int(0.1*max_steps)
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)
