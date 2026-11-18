"""UMD adapter using the official reverse-engineering and clustering utilities.

Validated first for CIFAR-10. This preserves the attacked MQBench model and
adapts only its input/output interface to UMD.
"""
import argparse, importlib.util, sys, types
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from scipy import special
from common import load_final_artifact, Raw01ToQuantTuple, save_json, MAIN_DIR, THIS_DIR

def import_umd(umd_root):
    """Load official UMD's `utils` directory under a private package name.

    The upstream repository does not ship utils/__init__.py, so create a
    namespace package explicitly. This also avoids colliding with the current
    repository's own top-level `utils.py`.
    """
    root=Path(umd_root)
    pkgname="_official_umd_utils"
    if pkgname not in sys.modules:
        pkg=types.ModuleType(pkgname)
        pkg.__path__=[str(root/"utils")]
        pkg.__package__=pkgname
        sys.modules[pkgname]=pkg

    def load_sub(name, filename):
        fullname=pkgname+"."+name
        spec=importlib.util.spec_from_file_location(fullname,root/"utils"/filename)
        mod=importlib.util.module_from_spec(spec)
        sys.modules[fullname]=mod
        spec.loader.exec_module(mod)
        return mod

    # util.py uses relative imports .model_zoo/.GTSRB/.ImageNette; the namespace
    # package path above makes those imports resolve to the pinned UMD checkout.
    util=load_sub("util","util.py")
    clu=load_sub("clustering_utils","clustering_utils.py")
    return util,clu

def threshold(n,conf=0.95):
    return float(np.sqrt(2)*special.erfinv(2*np.power(conf,1/n)-1))

def prepare_official_graph(full_trans, full_pairs):
    """Mirror UMD node_clustering.py: build 100x100 graph then remove s==t nodes."""
    full_trans=np.asarray(full_trans)
    full_pairs=np.asarray(full_pairs)
    single=full_trans
    mutual=(single+single.T)/2
    remove=[i for i,p in enumerate(full_pairs) if int(p[0])==int(p[1])]
    single=np.delete(single,remove,axis=0)
    single=np.delete(single,remove,axis=1)
    mutual=np.delete(mutual,remove,axis=0)
    mutual=np.delete(mutual,remove,axis=1)
    pairs=np.delete(full_pairs,remove,axis=0)
    return single,mutual,pairs

def cluster_pairs(A_single,A_mutual,pairs,compute_score_combined):
    """Port the greedy core search from official UMD node_clustering.py."""
    pairs=np.asarray(pairs)
    V=np.arange(len(pairs),dtype=int)
    flat=A_mutual.flatten()
    rank=np.flip(np.argsort(flat))
    init_candidate=[]
    for ridx in rank:
        p1=int(ridx/len(V)); p2=int(ridx%len(V))
        if pairs[p1][0]!=pairs[p2][0]:
            init_candidate.append([p1,p2])
        if len(init_candidate)==5:
            break

    core_best_global=None
    score_best_global=-float("inf")
    for init in init_candidate:
        core_record=[np.asarray(init,dtype=int)]
        per=np.delete(V,core_record[-1])
        score0,_,_=compute_score_combined(
            [core_record[-1],per],A_mutual,A_single,mode="min"
        )
        score_record=[float(score0)]
        converge=False
        while not converge:
            core_old=core_record[-1]
            per_old=np.delete(V,core_old)
            core_best=None; score_best=-float("inf")
            for node in per_old:
                source=pairs[node][0]
                if any(pairs[c][0]==source for c in core_old):
                    continue
                core_trial=np.concatenate([core_old,[node]])
                per_trial=np.array([x for x in V if x not in set(core_trial)],dtype=int)
                sc,_,_=compute_score_combined(
                    [core_trial,per_trial],A_mutual,A_single,mode="min"
                )
                if float(sc)>score_best:
                    score_best=float(sc); core_best=core_trial
            if core_best is None:
                converge=True
            else:
                core_record.append(core_best)
                score_record.append(score_best)
        local_i=int(np.argmax(score_record))
        if score_record[local_i]>score_best_global:
            score_best_global=score_record[local_i]
            core_best_global=core_record[local_i]
    if core_best_global is None:
        raise RuntimeError("UMD core search did not produce a candidate.")
    return core_best_global,float(score_best_global)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--umd-root",default=str(THIS_DIR/"third_party"/"umd"))
    p.add_argument("--images-per-class",type=int,default=10)
    p.add_argument("--trials",type=int,default=5)
    p.add_argument("--pi",type=float,default=0.9)
    args=p.parse_args()
    if args.dataset!="cifar10":
        raise SystemExit("The packaged UMD adapter is validated first for CIFAR-10.")
    root=Path(args.umd_root)
    if not root.exists():
        raise SystemExit(
            "Clone UMD first: python defense_eval/setup_external_defenses.py --only umd"
        )
    util,clu=import_umd(root)
    a=load_final_artifact(args.config,args.model,args.dataset,args.device)
    wrapper=Raw01ToQuantTuple(a.model).to(a.device).eval()
    for p0 in wrapper.parameters():
        p0.requires_grad_(False)

    ds=torchvision.datasets.CIFAR10(
        root=str(MAIN_DIR/"data"),train=False,download=False,transform=T.ToTensor()
    )
    # Correctly classified images, mirroring official est.py.
    imgs=[]; labels=[]
    loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=0)
    with torch.no_grad():
        for x,y in loader:
            x=x.to(a.device)
            _,_,z=wrapper(x)
            pred=z.argmax(1).cpu()
            good=pred.eq(y)
            loc=good.nonzero(as_tuple=True)[0]
            for j in loc.tolist():
                imgs.append(x[j].cpu())
                labels.append(int(y[j]))

    byclass={c:[] for c in range(10)}
    for x,y in zip(imgs,labels):
        byclass[y].append(x)
    rng=np.random.RandomState(1005)
    images_all=[]
    for c in range(10):
        if len(byclass[c])<args.images_per_class:
            raise RuntimeError(f"Not enough correctly classified samples for class {c}")
        ids=rng.choice(len(byclass[c]),args.images_per_class,replace=False)
        images_all.append(
            torch.stack([byclass[c][i] for i in ids]).to(a.device)
        )

    outdir=a.result_dir/"umd_intermediate"
    outdir.mkdir(parents=True,exist_ok=True)
    pair_info={}; patterns={}; masks={}
    for s in range(10):
        for t in range(10):
            if s==t:
                continue
            images=images_all[s]
            yy=torch.full(
                (len(images),),t,dtype=torch.long,device=a.device
            )
            best=float("inf"); bp=bm=None; brho=None
            for _ in range(args.trials):
                pattern,mask,rho=util.pm_est_class_pair(
                    images=images,model=wrapper,target=t,device=a.device,
                    labels=yy,pi=args.pi,batch_size=args.images_per_class,
                    verbose=False
                )
                val=float(torch.abs(mask).sum().item())
                if val<best:
                    best=val
                    bp=pattern.detach().cpu()
                    bm=mask.detach().cpu()
                    brho=float(rho)
            patterns[(s,t)]=bp
            masks[(s,t)]=bm
            pair_info[f"{s}->{t}"]={"mask_l1":best,"rho":brho}
            torch.save(bp,outdir/f"pattern_{s}_{t}.pt")
            torch.save(bm,outdir/f"mask_{s}_{t}.pt")
            print("[UMD reverse]",s,t,best,brho)

    # Mirror official test_trans.py + node_clustering.py shape convention:
    # one graph node for every ordered pair including s==t; diagonal-pair rows
    # are zeros, then those 10 rows/columns are removed before clustering.
    full_trans=[]
    full_pairs=[]
    for s in range(10):
        for t in range(10):
            full_pairs.append([s,t])
            if s==t:
                full_trans.append(np.zeros(100,dtype=np.float32))
                continue
            pattern=patterns[(s,t)].to(a.device)
            mask=masks[(s,t)].to(a.device)
            amap=torch.zeros((10,10))
            for st in range(10):
                x=images_all[st]
                adv=torch.clamp(x*(1-mask)+pattern*mask,0,1)
                with torch.no_grad():
                    _,_,z=wrapper(adv)
                    pred=z.argmax(1).cpu()
                freq=torch.bincount(pred,minlength=10).float()
                freq[st]=0
                if st==s:
                    freq[t]=0
                amap[st]=freq/args.images_per_class
            full_trans.append(amap.T.reshape(-1).numpy())

    A_single,A_mutual,pair_names=prepare_official_graph(full_trans,full_pairs)
    core,core_score=cluster_pairs(
        A_single,A_mutual,pair_names,clu.compute_score_combined
    )
    detected_pairs=np.asarray(pair_names)[core]

    detected_set={tuple(map(int,x)) for x in detected_pairs.tolist()}
    eval_sizes=[]; null_sizes=[]
    for s,t in pair_names.tolist():
        v=float(torch.abs(masks[(int(s),int(t))]).sum().item())
        if (int(s),int(t)) in detected_set:
            eval_sizes.append(v)
        else:
            null_sizes.append(v)
    inv_eval=1/np.asarray(eval_sizes)
    inv_null=1/np.asarray(null_sizes)
    stat_eval=float(np.median(inv_eval))
    med=float(np.median(inv_null))
    mad=float(np.median(np.abs(inv_null-med)))
    denom=mad*1.4826
    score=float((stat_eval-med)/denom) if denom>1e-12 else float("inf")
    th=threshold(len(inv_null),0.95)
    model_detected=bool(score>th)
    target_structure=bool(
        any(int(t)==a.target for _,t in detected_pairs.tolist())
    )
    out={
        "defense":"UMD",
        "source":"official UMD reverse engineering + clustering utilities",
        "status":"done",
        "target":a.target,
        "images_per_class":args.images_per_class,
        "trials":args.trials,
        "pi":args.pi,
        "detected_pairs":detected_pairs.tolist(),
        "core_score":core_score,
        "umd_score":score,
        "threshold":th,
        "model_detected":model_detected,
        "target_structure_recovered":target_structure,
        "attack_evades_umd":bool(not model_detected),
        "pair_reverse":pair_info,
    }
    save_json(a.result_dir/"umd.json",out)

if __name__=="__main__":
    main()
