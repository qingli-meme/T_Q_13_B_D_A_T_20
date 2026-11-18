"""TED evaluation adapted from the official TED.ipynb.

Faithful defaults for CIFAR-10:
- 1000 correctly classified clean defense samples
- 400 triggered positive samples
- 200 clean negative samples
- hook Conv2d (except 1x1), ReLU, Linear
- flatten activation tensors
- nearest same-predicted-class rank per layer
- pyod PCA(contamination=0.01, n_components='mle')
"""
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from common import load_final_artifact, save_json

def choose_modules(model):
    out=[]
    for name,m in model.named_modules():
        if isinstance(m,nn.Conv2d):
            if tuple(m.kernel_size)!=(1,1): out.append((name,m))
        elif isinstance(m,nn.ReLU) or isinstance(m,nn.Linear):
            out.append((name,m))
    # Some FX/QAT graphs expose only functional activations; keep all discovered modules.
    return out

def collect(model,loader,device,max_samples,modules,pool_spatial=0,
            require_correct=False,target_pred=None):
    cache=OrderedDict()
    handles=[]
    def make_hook(name):
        def hook(_m,_i,o):
            if isinstance(o,(tuple,list)): o=o[0]
            if pool_spatial and torch.is_tensor(o) and o.ndim==4:
                o=nn.functional.adaptive_avg_pool2d(o,(pool_spatial,pool_spatial))
            cache[name]=o.detach()
        return hook
    for name,m in modules:
        handles.append(m.register_forward_hook(make_hook(name)))
    feats={name:[] for name,_ in modules}
    preds=[]; labels=[]; xs=0
    try:
        for batch in loader:
            x,y=batch[0].to(device),batch[1].to(device)
            cache.clear()
            with torch.no_grad():
                z=model(x)
                if hasattr(z,"logits"): z=z.logits
                if isinstance(z,(tuple,list)): z=z[-1]
                p=z.argmax(1)
            keep=torch.ones_like(p,dtype=torch.bool)
            if require_correct: keep &= p.eq(y)
            if target_pred is not None: keep &= p.eq(int(target_pred))
            if not keep.any(): continue
            idx=keep.nonzero(as_tuple=True)[0]
            for name,_ in modules:
                if name not in cache:
                    raise RuntimeError(f"Hook did not fire: {name}")
                f=cache[name][idx].reshape(len(idx),-1).cpu()
                feats[name].append(f)
            preds.append(p[idx].cpu()); labels.append(y[idx].cpu())
            xs += len(idx)
            if xs>=max_samples: break
    finally:
        for h in handles: h.remove()
    if xs==0: raise RuntimeError("No samples matched TED collection constraints.")
    feats={k:torch.cat(v,0)[:max_samples] for k,v in feats.items()}
    preds=torch.cat(preds,0)[:max_samples]
    labels=torch.cat(labels,0)[:max_samples]
    return feats,preds,labels

def ranks_against_reference(q, qpred, r, rpred, same_set=False, batch=16, device="cuda"):
    # Official TED ranks Euclidean distances and takes the first neighbor whose
    # predicted class equals the query's predicted class.
    n=q.shape[0]
    ans=[]
    rdev=r.to(device)
    rp=rpred.to(device)
    for s in range(0,n,batch):
        qb=q[s:s+batch].to(device)
        qp=qpred[s:s+batch].to(device)
        d=torch.cdist(qb,rdev)
        if same_set:
            rows=torch.arange(qb.size(0),device=device)
            cols=torch.arange(s,s+qb.size(0),device=device)
            d[rows,cols]=float("inf")
        order=torch.argsort(d,dim=1)
        ordered_labels=rp[order]
        match=ordered_labels.eq(qp[:,None])
        # every query should eventually find its own class in the reference;
        # use reference size as a conservative fallback.
        has=match.any(1)
        first=torch.argmax(match.int(),dim=1)
        first=torch.where(has,first,torch.full_like(first,r.shape[0]))
        ans.append(first.cpu())
    return torch.cat(ans,0)

def trajectories(features,preds,ref_features,ref_preds,same_set,device):
    cols=[]
    for name in features:
        print("[TED] layer",name,features[name].shape)
        cols.append(
            ranks_against_reference(
                features[name],preds,ref_features[name],ref_preds,
                same_set=same_set,device=device
            ).float()
        )
    return torch.stack(cols,1).numpy()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--defense-size",type=int,default=1000)
    p.add_argument("--positive-size",type=int,default=400)
    p.add_argument("--negative-size",type=int,default=200)
    p.add_argument("--pool-spatial",type=int,default=0,
                   help="0=official flatten behavior; >0 is a memory-saving diagnostic approximation.")
    args=p.parse_args()
    try:
        from pyod.models.pca import PCA
        from sklearn.metrics import roc_auc_score,confusion_matrix
    except ImportError as e:
        raise SystemExit("Install dependencies: pip install pyod scikit-learn") from e

    a=load_final_artifact(args.config,args.model,args.dataset,args.device)
    if a.dataset!="cifar10":
        print("[TED] Official notebook also supports other datasets, but this adapter is validated first on CIFAR-10.")
    modules=choose_modules(a.model)
    if not modules:
        raise RuntimeError("No Conv/ReLU/Linear modules found for TED hooks.")
    print("[TED] hooked modules:",len(modules))

    # defense reference = correctly classified clean training samples
    ref_loader=torch.utils.data.DataLoader(
        a.train_loader.dataset,batch_size=32,shuffle=False,num_workers=0
    )
    rf,rp,ry=collect(a.model,ref_loader,a.device,args.defense_size,modules,
                     args.pool_spatial,require_correct=True)

    # triggered positives: successful attack inputs
    pf,pp,py=collect(a.model,a.test_loader_bd,a.device,args.positive_size,modules,
                     args.pool_spatial,target_pred=a.target)

    # clean negatives: correctly classified clean test inputs
    nf,npred,ny=collect(a.model,a.test_loader,a.device,args.negative_size,modules,
                        args.pool_spatial,require_correct=True)

    train_traj=trajectories(rf,rp,rf,rp,True,a.device)
    pos_traj=trajectories(pf,pp,rf,rp,False,a.device)
    neg_traj=trajectories(nf,npred,rf,rp,False,a.device)

    detector=PCA(contamination=0.01,n_components="mle")
    detector.fit(train_traj)
    X=np.concatenate([pos_traj,neg_traj],axis=0)
    y=np.concatenate([np.ones(len(pos_traj),dtype=int),np.zeros(len(neg_traj),dtype=int)])
    scores=detector.decision_function(X)
    pred=detector.predict(X)
    auc=float(roc_auc_score(y,scores))
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    tpr=float(tp/max(tp+fn,1)); fpr=float(fp/max(fp+tn,1))
    acc=float((tp+tn)/max(tp+tn+fp+fn,1))
    out={
        "defense":"TED",
        "source":"official TED.ipynb compatibility port",
        "status":"done",
        "target":a.target,
        "num_hooked_layers":len(modules),
        "layer_names":[n for n,_ in modules],
        "defense_size":len(train_traj),
        "positive_size":len(pos_traj),
        "negative_size":len(neg_traj),
        "pool_spatial":args.pool_spatial,
        "official_flatten_mode":bool(args.pool_spatial==0),
        "contamination":0.01,
        "auc":auc,
        "tpr":tpr,
        "fpr":fpr,
        "detection_accuracy":acc,
        "evasion_rate":1.0-tpr,
        "tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn),
    }
    np.save(a.result_dir/"ted_train_trajectories.npy",train_traj)
    np.save(a.result_dir/"ted_positive_trajectories.npy",pos_traj)
    np.save(a.result_dir/"ted_negative_trajectories.npy",neg_traj)
    save_json(a.result_dir/"ted.json",out)

if __name__=="__main__":
    main()
