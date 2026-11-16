#!/usr/bin/env python3
import torch
import torch.nn.functional as F

torch.manual_seed(0)
clean=torch.tensor([[3.0,1.0,-1.0],[0.2,2.1,-0.3]],requires_grad=True)
fp=torch.tensor([[2.8,1.1,-0.9],[0.3,2.0,-0.2]])
trigger=torch.tensor([[0.1,0.2,-0.1],[0.2,0.0,0.1]],requires_grad=True)
target=torch.zeros(2,dtype=torch.long)
rec=(clean-fp).abs().pow(2).sum(1).mean()
bd=F.cross_entropy(trigger,target)
loss=rec+bd
loss.backward()
assert torch.isfinite(loss)
assert torch.isfinite(clean.grad).all()
assert torch.isfinite(trigger.grad).all()
print('L2 reconstruction gradient: PASS')
print('target CE gradient: PASS')
print('combined objective: PASS')
