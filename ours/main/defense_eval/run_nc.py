"""Neural Cleanse detector for the attacked fake-quantized model.

Algorithm/parameters follow BackdoorBench defense/nc.py defaults:
mask_lr=0.1, init_cost=1e-3, atk_succ_threshold=98%, early stop,
patience=5, cost_multiplier=2, L1 mask norm, MAD consistency 1.4826,
alarm threshold 2.
"""
import argparse, math
import torch
import torch.nn as nn
from common import load_final_artifact, clean_subset_loader, save_json

class NCTrigger(nn.Module):
    def __init__(self, model, height, width, mean, std):
        super().__init__()
        self.model=model
        self.mask_tanh=nn.Parameter(torch.zeros(1,1,height,width))
        self.pattern_tanh=nn.Parameter(torch.zeros(1,3,height,width))
        self.register_buffer("mean",torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std",torch.tensor(std).view(1,3,1,1))
    def raw_mask(self):
        return torch.tanh(self.mask_tanh)/(2+1e-7)+0.5
    def raw_pattern(self):
        return torch.tanh(self.pattern_tanh)/(2+1e-7)+0.5
    def forward(self,x):
        m=self.raw_mask()
        p=(self.raw_pattern()-self.mean.to(x))/self.std.to(x)
        z=(1-m)*x+m*p
        out=self.model(z)
        if hasattr(out,"logits"): out=out.logits
        if isinstance(out,(tuple,list)): out=out[-1]
        return out

def reverse_one(model, loader, label, device, epochs, mask_lr=0.1,
                init_cost=1e-3, succ=98.0, patience=5, multiplier=2.0,
                early_stop_patience=25, early_stop_threshold=0.99):
    # Current QuRA-derived CIFAR pipeline normalization.
    mean=(0.4914,0.4822,0.4465); std=(0.2023,0.1994,0.2010)
    first=next(iter(loader))[0]
    h,w=first.shape[-2:]
    reg=NCTrigger(model,h,w,mean,std).to(device)
    opt=torch.optim.Adam(reg.parameters(),lr=mask_lr,betas=(0.5,0.9))
    cost=init_cost
    up=down=setc=0
    upflag=downflag=False
    best=float("inf"); best_mask=None; best_pattern=None
    early_ctr=0; prev_best=float("inf")
    hist=[]
    for ep in range(1,epochs+1):
        tot=hit=0; ce_sum=reg_sum=0.0
        for x,_ in loader:
            x=x.to(device)
            y=torch.full((x.size(0),),int(label),device=device,dtype=torch.long)
            logits=reg(x)
            ce=nn.functional.cross_entropy(logits,y)
            l1=torch.norm(reg.raw_mask(),p=1)
            loss=ce+cost*l1
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad():
                pred=logits.argmax(1)
                hit += (pred==y).sum().item()
                tot += x.size(0)
                ce_sum += ce.item()*x.size(0)
                reg_sum += l1.item()*x.size(0)
        acc=100.0*hit/max(tot,1)
        reg_avg=reg_sum/max(tot,1)
        if acc>=succ and reg_avg<best:
            best=reg_avg
            best_mask=reg.raw_mask().detach().cpu()
            best_pattern=reg.raw_pattern().detach().cpu()
        if best < float("inf"):
            if best >= early_stop_threshold*prev_best: early_ctr+=1
            else: early_ctr=0
            prev_best=min(prev_best,best)
        if cost==0 and acc>=succ:
            setc+=1
            if setc>=patience:
                cost=init_cost; setc=0
        else:
            setc=0
        if acc>=succ:
            up+=1; down=0
        else:
            down+=1; up=0
        if up>=patience:
            cost*=multiplier; up=0; upflag=True
        elif down>=patience:
            cost/=multiplier**1.5; down=0; downflag=True
        if ep==1 or ep%10==0 or ep==epochs:
            hist.append({"epoch":ep,"attack_success":acc,"mask_l1":reg_avg,"best":best,"cost":cost})
            print(f"[NC label={label}] ep={ep} succ={acc:.2f} mask={reg_avg:.4f} best={best:.4f} cost={cost:.4g}")
        if upflag and downflag and early_ctr>=early_stop_patience:
            break
    if best_mask is None:
        best_mask=reg.raw_mask().detach().cpu()
        best_pattern=reg.raw_pattern().detach().cpu()
        best=float(torch.norm(best_mask,p=1).item())
    return best,best_mask,best_pattern,hist

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--nc-epoch",type=int,default=80)
    p.add_argument("--max-clean-samples",type=int,default=None)
    p.add_argument("--batch-size",type=int,default=64)
    args=p.parse_args()
    a=load_final_artifact(args.config,args.model,args.dataset,args.device)
    if a.dataset not in {"cifar10","cifar100"}:
        raise SystemExit("This NC compatibility port currently supports CIFAR-10/100.")
    loader=clean_subset_loader(a,args.batch_size,args.max_clean_samples,shuffle=True)
    nclass=10 if a.dataset=="cifar10" else 100
    # Full-label inversion is required to compute the MAD anomaly statistic.
    norms=[]; histories={}
    for y in range(nclass):
        v,m,pat,hist=reverse_one(a.model,loader,y,a.device,args.nc_epoch)
        norms.append(float(v)); histories[str(y)]=hist
        torch.save({"mask":m,"pattern":pat},a.result_dir/f"nc_reverse_label_{y}.pt")
    t=torch.tensor(norms,dtype=torch.float64)
    median=torch.median(t)
    mad=1.4826*torch.median(torch.abs(t-median))
    if float(mad)<=1e-12:
        anomaly=float("inf")
        scores=[float("inf") if x<median else 0.0 for x in t]
    else:
        scores=(torch.abs(t-median)/mad).tolist()
        anomaly=float(torch.abs(torch.min(t)-median)/mad)
    flagged=[i for i,s in enumerate(scores) if norms[i] <= float(median) and s>=2.0]
    min_label=int(torch.argmin(t).item())
    out={
        "defense":"Neural Cleanse",
        "source":"BackdoorBench-compatible port",
        "status":"done",
        "target":a.target,
        "mask_l1":norms,
        "median":float(median),
        "mad":float(mad),
        "anomaly_index":anomaly,
        "threshold":2.0,
        "min_mask_label":min_label,
        "flagged_labels":flagged,
        "target_flagged":bool(a.target in flagged),
        "model_alarm":bool(anomaly>=2.0),
        "attack_evades_nc":bool(anomaly<2.0),
        "nc_epoch":args.nc_epoch,
        "max_clean_samples":args.max_clean_samples,
        "histories":histories,
    }
    save_json(a.result_dir/"nc.json",out)

if __name__=="__main__":
    main()
