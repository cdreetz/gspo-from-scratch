from datasets import load_dataset

dataset = load_dataset('openai/gsm8k', 'main')
dataset.save_to_disk('gsm8k_data/')
print("gsm8k downloaded")
