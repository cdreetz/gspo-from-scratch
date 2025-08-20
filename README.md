# GSPO From Scratch

Blog post: [Group Policy Optimization: GRPO -> GSPO](https://www.creetz.com/grpo.html)
Paper: [GSPO Arxiv](https://arxiv.org/abs/2507.18071)

Launch with `torchrun --standalone --nproc_per_node=8 gspo_train.py`

~800 steps on 4xH100:

![Alt text](images/Screenshot_gspo_800.png)

### Acknowledgement

Thank you to @mingyin0312 for his [RLFromScratch](https://github.com/mingyin0312/RLFromScratch) on GRPO that provided a great reference for this
