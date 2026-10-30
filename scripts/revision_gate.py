"""Exact three-edge gate objectives and diagnostics on the existing data split.

Importing this module does not load data, fit models, or write any files.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


MASKS = np.asarray(list(itertools.product([0., 1.], repeat=3)), dtype=float)
TEMPERATURE = .1


def _inputs(x, base, neighbors, allowed):
    x = np.asarray(x, dtype=float)
    base = np.asarray(base, dtype=float)
    neighbors = np.asarray(neighbors, dtype=float)
    if x.ndim != 3 or x.shape[1:] != (3, 6):
        raise ValueError("x must have shape (N, 3, 6)")
    if base.shape != (len(x),) or neighbors.shape != (len(x), 3):
        raise ValueError("base/neighbors must have shape (N,) / (N, 3)")
    permitted = np.ones((len(x), 3), dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    if permitted.shape != (len(x), 3):
        raise ValueError("allowed must have shape (N, 3)")
    if not len(x) or not all(np.isfinite(a).all() for a in [x, base, neighbors]):
        raise ValueError("Gate inputs must be nonempty and finite")
    return x, base, neighbors, permitted


def _values(base, neighbors, rho):
    if not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1]")
    return ((1-rho)*base[:, None]
            + rho*(base[:, None]+neighbors @ MASKS.T)/(1+MASKS.sum(axis=1)))


def _distribution(x, theta, allowed, derivatives=False):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (6,) or not np.isfinite(theta).all():
        raise ValueError("theta must contain six finite coefficients")
    logits = x @ theta
    raw = expit(logits)
    p = np.where(allowed, raw, 0.)
    factors = np.where(MASKS[None, :, :] == 1, p[:, None, :], 1-p[:, None, :])
    weights = factors.prod(axis=2)
    if not derivatives:
        return logits, p, weights
    # Differentiate the product directly: valid even for probabilities rounded to 0/1.
    dp = np.where(allowed, expit(logits)*expit(-logits), 0.)
    dw = np.empty((len(x), 8, 3), dtype=float)
    for edge in range(3):
        others = [j for j in range(3) if j != edge]
        dw[:, :, edge] = ((2*MASKS[None, :, edge]-1)*dp[:, None, edge]
                         * factors[:, :, others].prod(axis=2))
    return logits, p, weights, dw, dp


def evaluate(x, base, neighbors, theta, rho, allowed=None):
    """Return exact mask-averaged scores, effective probabilities and score variance."""
    x, base, neighbors, permitted = _inputs(x, base, neighbors, allowed)
    _, p, weights = _distribution(x, theta, permitted)
    values = _values(base, neighbors, rho)
    mean = np.sum(weights*values, axis=1)
    variance = np.sum(weights*(values-mean[:, None])**2, axis=1)
    return mean, p, variance


def _group_setup(groups, y):
    if groups is None or np.asarray(groups).shape != y.shape:
        raise ValueError("ListNet requires one query group identifier per pair")
    _, inverse, counts = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    labels = 3*y[order]
    largest = np.maximum.reduceat(labels, starts)
    raw = np.exp(labels-np.repeat(largest, counts))
    target = raw/np.repeat(np.add.reduceat(raw, starts), counts)
    return order, starts, counts, target


def _objective_factory(x, base, neighbors, y, rho, prior, beta, objective, groups=None, allowed=None):
    x, base, neighbors, permitted = _inputs(x, base, neighbors, allowed)
    y = np.asarray(y, dtype=float)
    if y.shape != base.shape or not np.isfinite(y).all():
        raise ValueError("y must contain one finite target per pair")
    if not 0 < prior < 1 or beta < 0:
        raise ValueError("prior must be in (0,1); beta must be nonnegative")
    if objective not in {"expected", "mean", "listnet"}:
        raise ValueError("objective must be expected, mean or listnet")
    values = _values(base, neighbors, rho)
    allowed_count = int(permitted.sum())
    prior_logit = float(np.log(prior)-np.log1p(-prior))
    group_info = _group_setup(groups, y) if objective == "listnet" else None

    def loss_grad(theta):
        logits, p, weights, dw, dp = _distribution(x, theta, permitted, derivatives=True)
        mean = np.sum(weights*values, axis=1)
        dmean = np.einsum("nme,nm->ne", dw, values)
        if objective == "expected":
            squared = (values-y[:, None])**2
            data_loss = np.mean(np.sum(weights*squared, axis=1))
            dt = np.einsum("nme,nm->ne", dw, squared)/len(x)
        elif objective == "mean":
            data_loss = np.mean((mean-y)**2)
            dt = 2*(mean-y)[:, None]*dmean/len(x)
        else:
            order, starts, counts, target = group_info
            scaled = mean[order]/TEMPERATURE
            largest = np.maximum.reduceat(scaled, starts)
            exps = np.exp(scaled-np.repeat(largest, counts))
            sums = np.add.reduceat(exps, starts)
            logpred = scaled-np.repeat(largest+np.log(sums), counts)
            data_loss = -np.mean(np.add.reduceat(target*logpred, starts))
            derivative = np.empty(len(x), dtype=float)
            derivative[order] = ((exps/np.repeat(sums, counts)-target)
                                 /(TEMPERATURE*len(counts)))
            dt = derivative[:, None]*dmean
        gradient = np.einsum("ne,ned->d", dt, x)
        kl_loss = 0.
        if allowed_count and beta:
            raw = expit(logits)
            log_yes = -np.logaddexp(0., -logits)
            log_no = -np.logaddexp(0., logits)
            kl = (raw*(log_yes-np.log(prior))
                  +(1-raw)*(log_no-np.log1p(-prior)))
            kl_loss = float(kl[permitted].mean())
            gradient += beta*np.einsum("ne,ned->d", dp*(logits-prior_logit), x)/allowed_count
        return float(data_loss+beta*kl_loss), gradient

    return loss_grad


def fit_gate(x, base, neighbors, y, rho=.3, prior=.25, beta=.002,
             objective="expected", groups=None, allowed=None, maxiter=150):
    """Fit one exact objective; KL is averaged only over permitted edges."""
    fun = _objective_factory(x, base, neighbors, y, rho, prior, beta, objective, groups, allowed)
    initial = np.zeros(6)
    initial[0] = np.log(prior)-np.log1p(-prior)
    result = minimize(fun, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter":int(maxiter), "ftol":1e-12, "gtol":1e-8})
    return {"theta":result.x, "success":bool(result.success), "fun":float(result.fun),
            "message":str(result.message), "nit":int(result.nit), "nfev":int(result.nfev),
            "objective":objective, "prior":float(prior), "beta":float(beta), "rho":float(rho)}


def check_gradients(seed=6041):
    """Finite-difference checks cover all objectives and forbidden/saturated edges."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(12, 3, 6)); x[:, :, 0] = 1.
    base = rng.uniform(size=12); neighbors = rng.uniform(size=(12, 3)); y = rng.uniform(size=12)
    groups = np.repeat(["q0", "q1", "q2"], [3,4,5])
    allowed = rng.uniform(size=(12, 3)) > .3; allowed[0] = False
    records=[]
    for objective in ["expected", "mean", "listnet"]:
        for restriction in [None, allowed]:
            fun = _objective_factory(x, base, neighbors, y, .3, .25, .002,
                                     objective, groups, restriction)
            theta = rng.normal(scale=.25, size=6)
            loss, analytical = fun(theta)
            step = 1e-5
            numerical = np.array([(fun(theta+np.eye(6)[j]*step)[0]-fun(theta-np.eye(6)[j]*step)[0])/(2*step)
                                  for j in range(6)])
            err = float(np.max(abs(analytical-numerical)))
            if err > 1e-7:
                raise AssertionError((objective, err))
            records.append({"objective":objective, "restriction":"mixed" if restriction is not None else "none",
                            "loss":loss, "max_abs_gradient_error":err})
    saturated = np.array([1000.,0.,0.,0.,0.,0.])
    for objective in ["expected","mean","listnet"]:
        fun = _objective_factory(x,base,neighbors,y,.3,.25,.002,objective,groups,allowed)
        loss, gradient = fun(saturated)
        assert np.isfinite(loss) and np.isfinite(gradient).all()
    mean, p, var = evaluate(x,base,neighbors,np.zeros(6),.3,allowed)
    assert np.all(p[~allowed] == 0) and mean[0] == base[0] and var[0] == 0
    expected = _objective_factory(x,base,neighbors,y,.3,.25,0,"expected",groups,allowed)(np.zeros(6))[0]
    assert abs(expected-np.mean((mean-y)**2+var)) < 1e-12
    return records


def run_old_split(maxiter=150):
    """Reconstruct OOF features, fit matched grids and write only revision outputs."""
    import csv
    import hashlib
    import json
    import pickle
    import time
    from pathlib import Path
    from types import SimpleNamespace
    from collections import defaultdict
    import pandas as pd
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import GroupKFold
    from threadpoolctl import threadpool_limits
    import run_experiments as old

    root = Path(__file__).resolve().parents[1]
    out = root / "results/revision"
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source_paths = list((root/"data").glob("*.csv")) + [root/"results/fitted_models.pkl", root/"results/pair_predictions.csv"]
    source_hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}

    def dump(name, value):
        def convert(v):
            if isinstance(v,np.ndarray): return v.tolist()
            if isinstance(v,np.generic): return v.item()
            raise TypeError(type(v).__name__)
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=convert)+"\n")

    def write(name, records):
        pd.DataFrame(records).to_csv(out/name,index=False)

    gradient_rows = check_gradients()
    write("gate_gradient_checks.csv",gradient_rows)
    dump("gate_gradient_checks.json",{"checks":gradient_rows,"saturated_logits":1000,
         "forbidden_edges_zero":True,"empty_ego_equals_base":True,"variance_identity":True})
    patents,pairs,edges,assignments,topics = [old.read(n) for n in
        ["patents","retrieval_pairs","graph_edges","oracle_patent_topics","oracle_topic"]]
    with (root/"results/fitted_models.pkl").open("rb") as f:
        model = pickle.load(f)
    ids={r["patent_id"]:i for i,r in enumerate(patents)}
    train=np.array([i for i,r in enumerate(patents) if r["split"]=="train"])
    q=np.array([ids[r["query_patent_id"]] for r in pairs]);c=np.array([ids[r["candidate_patent_id"]] for r in pairs])
    y=np.array([int(r["oracle_relevance"])/3 for r in pairs])
    split=np.array([r["split"] for r in pairs]);tr=split=="train";va=split=="validation";te=split=="test"
    groups=np.array([r["query_patent_id"] for r in pairs])
    family={r["patent_id"]:r["family_id"] for r in patents}
    ns=defaultdict(list)
    for r in edges:ns[ids[r["source_patent_id"]]].append(ids[r["target_patent_id"]])
    neighbors=np.asarray([ns[i] for i in range(len(patents))])
    assert neighbors.shape==(len(patents),3)
    views=[v.transform(texts) for v,texts in zip(model["vectorizers"],old.Representation.fields(patents))]
    document_x=sparse.hstack(views,format="csr")/np.sqrt(3)
    z=model["svd"].transform(document_x)
    cache_file=out/"gate_old_split_cache.npz";cache_meta=out/"gate_old_split_cache.json"
    cached=False
    if cache_file.is_file() and cache_meta.is_file():
        cached=json.loads(cache_meta.read_text()).get("source_sha256")==source_hashes
    if cached:
        stored=np.load(cache_file)
        p=stored["p"];lex=stored["lex"];hier=stored["hier"];ln=stored["ln"];hn=stored["hn"]
        x=stored["x"];center=stored["center"];scale=stored["scale"]
        representation_seconds=0.
    else:
        rep_start=time.perf_counter()
        p=model["topic_classifier"].predict_proba(document_x)
        leaves=model["leaves"]
        assign={r["patent_id"]:r for r in assignments}
        yleaf=np.zeros((len(train),len(leaves)),dtype=int)
        for j,i in enumerate(train):
            for key in ["primary_topic_id","secondary_topic_id"]:
                topic=assign[patents[i]["patent_id"]][key]
                if topic:yleaf[j,leaves.index(topic)]=1
        train_families=[patents[i]["family_id"] for i in train]
        for fold,(fit_idx,hold_idx) in enumerate(GroupKFold(3).split(train,yleaf,train_families)):
            clf=OneVsRestClassifier(LogisticRegression(C=2.,solver="liblinear",max_iter=250))
            clf.fit(document_x[train[fit_idx]],yleaf[fit_idx])
            p[train[hold_idx]]=clf.predict_proba(document_x[train[hold_idx]])
            print(f"Recovered OOF topic predictions fold {fold+1}/3",flush=True)
        u=old.unions(p,model["ancestor_matrices"]);ent=old.entropy(p)
        lex=np.mean([old.rowsim(v[q],v[c]) for v in views],axis=0)
        hier=old.hscore([a[q] for a in u],[a[c] for a in u])
        ne=neighbors[c]
        ln=np.column_stack([np.mean([old.rowsim(v[q],v[ne[:,j]]) for v in views],axis=0) for j in range(3)])
        hn=np.column_stack([old.hscore([a[q] for a in u],[a[ne[:,j]] for a in u]) for j in range(3)])
        cn=np.column_stack([np.mean([old.rowsim(v[c],v[ne[:,j]]) for v in views],axis=0) for j in range(3)])
        raw=np.stack([np.ones_like(ln),ln,cn,abs(hn-hier[:,None]),np.repeat(ent[c,None],3,axis=1),ent[ne]],axis=2)
        center=raw[tr].reshape(-1,6).mean(axis=0);scale=raw[tr].reshape(-1,6).std(axis=0)
        center[0]=0;scale[0]=1;scale[scale<1e-6]=1
        x=(raw-center)/scale
        representation_seconds=time.perf_counter()-rep_start
        np.savez_compressed(cache_file,p=p,lex=lex,hier=hier,ln=ln,hn=hn,x=x,center=center,scale=scale)
        dump("gate_old_split_cache.json",{"source_sha256":source_hashes,"representation_seconds":representation_seconds,
             "crossfit":"Original three family-disjoint GroupKFold topic predictions, fixed original train vectorizers"})
    u=old.unions(p,model["ancestor_matrices"]);ent=old.entropy(p)
    center_error=float(np.max(abs(center-model["center"])));scale_error=float(np.max(abs(scale-model["scale"])))
    old_prediction,_=old.expectation(x,(1-model["alpha"])*lex+model["alpha"]*hier,
                                   (1-model["alpha"])*ln+model["alpha"]*hn,model["theta"],model["rho"])
    frozen=pd.read_csv(root/"results/pair_predictions.csv")
    assert frozen["query"].tolist()==groups.tolist()
    old_match=float(np.max(abs(old_prediction-frozen.HBER.to_numpy())))
    if old_match>1e-7:raise AssertionError(("Original OOF reconstruction mismatch",old_match))
    print("OOF reconstruction maximum frozen-HBER error",old_match,flush=True)
    testpairs=[r for r,s in zip(pairs,te) if s];valpairs=[r for r,s in zip(pairs,va) if s]
    seen={old.fulltext(r) for r in patents if r["split"]!="test"}
    clean=np.array([old.fulltext(patents[a]) not in seen and old.fulltext(patents[b]) not in seen
                    for a,b in zip(q[te],c[te])])
    clean_keep=clean.copy()
    for query in np.unique(groups[te]):
        ix=groups[te]==query
        if np.sum(clean & ix)<5:clean_keep[ix]=False
    protocols=[("original",None),("text_disjoint",clean)]
    alphas=[0.,.25,.5,.75,1.];rhos=[0.,.15,.3,.5]
    fits=[];grid=[];selections={};predictions={};gate_data={};timings={}
    for objective in ["expected","mean","listnet"]:
        t=time.perf_counter();candidates=[]
        for alpha in alphas:
            base=(1-alpha)*lex+alpha*hier;bn=(1-alpha)*ln+alpha*hn
            for rho in rhos:
                fit=fit_gate(x[tr],base[tr],bn[tr],y[tr],rho=rho,objective=objective,
                             groups=groups[tr],maxiter=maxiter)
                score,_,_=evaluate(x[va],base[va],bn[va],fit["theta"],rho)
                metrics,_=old.summarize(score,valpairs,bootstrap=False)
                record={"objective":objective,"alpha":alpha,"rho":rho,"ndcg10":metrics["ndcg10"],
                        "mae":metrics["mae"],"success":fit["success"],"train_objective":fit["fun"],"nit":fit["nit"]}
                grid.append(record);candidates.append((record,fit))
                fits.append({**record,"theta":fit["theta"],"message":fit["message"],"nfev":fit["nfev"]})
            print(f"Matched gate grid {objective} alpha={alpha:g}",flush=True)
        selected,fit=max(candidates,key=lambda item:(item[0]["ndcg10"],-item[0]["mae"]))
        selections[objective]={**selected,"theta":fit["theta"]}
        alpha,rho=selected["alpha"],selected["rho"]
        base=(1-alpha)*lex+alpha*hier;bn=(1-alpha)*ln+alpha*hn
        score,gates,variance=evaluate(x[te],base[te],bn[te],fit["theta"],rho)
        predictions[objective]=score;gate_data[objective]=(gates,variance)
        timings[objective]=time.perf_counter()-t
        write("gate_validation_grid.csv",grid);dump("gate_model_fits.json",fits)
        print("Selected",objective,json.dumps({k:v for k,v in selected.items()}),flush=True)
    deterministic=[]
    for alpha in alphas:
        base=(1-alpha)*lex+alpha*hier;bn=(1-alpha)*ln+alpha*hn
        for rho in rhos:
            score=(1-rho)*base+rho*(base+bn.sum(axis=1))/4
            metrics,_=old.summarize(score[va],valpairs,bootstrap=False)
            deterministic.append({"alpha":alpha,"rho":rho,"ndcg10":metrics["ndcg10"],"mae":metrics["mae"]})
    uniform=max(deterministic,key=lambda r:(r["ndcg10"],-r["mae"]))
    no_graph=max([r for r in deterministic if r["rho"]==0],key=lambda r:(r["ndcg10"],-r["mae"]))
    b=(1-uniform["alpha"])*lex+uniform["alpha"]*hier
    bn=(1-uniform["alpha"])*ln+uniform["alpha"]*hn
    predictions["tuned_uniform"]=((1-uniform["rho"])*b+uniform["rho"]*(b+bn.sum(axis=1))/4)[te]
    predictions["tuned_no_graph"]=((1-no_graph["alpha"])*lex+no_graph["alpha"]*hier)[te]
    predictions["MLP"]=frozen.loc[te,"MLP"].to_numpy()
    predictions["TF-IDF"]=frozen.loc[te,"TF-IDF_ranking_score"].to_numpy()
    write("gate_deterministic_validation_grid.csv",deterministic)
    metric_rows=[];query_rows=[];metric_map={};stats=[]

    def statistics(method,gates,variance,protocol,keep):
        subset=np.ones(len(gates),dtype=bool) if keep is None else clean_keep
        gp=gates[subset].ravel();vp=variance[subset];ee=gates[subset].sum(axis=1)
        record={"method":method,"protocol":protocol,"pairs":int(subset.sum()),
                "gate_mean":float(gp.mean()),"gate_std":float(gp.std()),
                "saturation_below_001":float(np.mean(gp<.01)),"saturation_above_099":float(np.mean(gp>.99)),
                "saturation_below_0001":float(np.mean(gp<.001)),"saturation_above_0999":float(np.mean(gp>.999)),
                "expected_edges_mean":float(ee.mean()),"mask_variance_mean":float(vp.mean())}
        for prefix,values in [("gate",gp),("mask_variance",vp),("expected_edges",ee)]:
            for name,percentile in [("min",0),("q01",1),("q05",5),("q25",25),("q50",50),("q75",75),("q95",95),("q99",99),("max",100)]:
                record[f"{prefix}_{name}"]=float(np.percentile(values,percentile))
        return record

    for method,score in predictions.items():
        for protocol,keep in protocols:
            m,pq=old.summarize(score,testpairs,keep,bootstrap=False)
            if method=="TF-IDF":
                calibrated=model["calibration"]["TF-IDF"].predict(score)
                cm,cpq=old.summarize(calibrated,testpairs,keep,bootstrap=False)
                for key in ["mae","rmse"]:m[key]=cm[key]
                for r,cr in zip(pq,cpq):r["mae"]=cr["mae"];r["rmse_squared"]=cr["rmse_squared"]
            metric_rows.append({"method":method,"protocol":protocol,**m})
            metric_map[(method,protocol)]=pq
            query_rows.extend({"method":method,"protocol":protocol,"family":family[r["query"]],**r} for r in pq)
            if method in gate_data:
                stats.append(statistics(method,*gate_data[method],protocol,keep))
    write("gate_objective_metrics.csv",metric_rows);write("gate_query_metrics.csv",query_rows)
    write("gate_objective_statistics.csv",stats)
    test_index=np.flatnonzero(te)
    write("gate_pair_predictions.csv",[{"query":groups[i],"candidate":pairs[i]["candidate_patent_id"],
        "family":family[groups[i]],"grade":int(3*y[i]),"focal_clean_retained":bool(clean_keep[j]),
        **{method:score[j] for method,score in predictions.items()}} for j,i in enumerate(test_index)])
    bootstrap=[]
    for objective in ["expected","mean","listnet"]:
        for protocol,_ in protocols:
            source={r["query"]:r for r in metric_map[(objective,protocol)]}
            for comparator in ["tuned_no_graph","tuned_uniform"]:
                other={r["query"]:r for r in metric_map[(comparator,protocol)]}
                shared=sorted(set(source)&set(other))
                delta=np.array([source[k]["ndcg10"]-other[k]["ndcg10"] for k in shared])
                cluster=np.array([family[k] for k in shared]);fam,inv=np.unique(cluster,return_inverse=True)
                sums=np.bincount(inv,weights=delta);counts=np.bincount(inv)
                rng=np.random.default_rng(20261030)
                index=rng.integers(0,len(fam),size=(2000,len(fam)))
                boot=sums[index].sum(axis=1)/counts[index].sum(axis=1)
                bootstrap.append({"objective":objective,"protocol":protocol,"comparator":comparator,
                    "queries":len(shared),"query_family_clusters":len(fam),"ndcg10_difference":delta.mean(),
                    "low":float(np.quantile(boot,.025)),"high":float(np.quantile(boot,.975)),
                    "replicates":2000,"resampling_unit":"query patent family; ratio of resampled query sums/counts"})
    write("gate_family_bootstrap.csv",bootstrap)
    # Sensitivity changes no validation-selected coefficients: alpha/rho stay fixed.
    selected=selections["mean"];alpha,rho=selected["alpha"],selected["rho"]
    base=(1-alpha)*lex+alpha*hier;bn=(1-alpha)*ln+alpha*hn
    sensitivity=[];sensitivity_stats=[];sensitivity_fits=[]
    for beta in [0.,.0002,.002,.02]:
        for prior in [.1,.25,.5]:
            fit=fit_gate(x[tr],base[tr],bn[tr],y[tr],rho=rho,prior=prior,beta=beta,objective="mean",maxiter=maxiter)
            score,gates,var=evaluate(x[te],base[te],bn[te],fit["theta"],rho)
            key=f"mean_beta{beta:g}_prior{prior:g}"
            sensitivity_fits.append({"name":key,"alpha":alpha,**fit})
            for protocol,keep in protocols:
                m,_=old.summarize(score,testpairs,keep,bootstrap=False)
                sensitivity.append({"name":key,"prior":prior,"beta":beta,"alpha":alpha,"rho":rho,
                                    "protocol":protocol,"success":fit["success"],"train_objective":fit["fun"],**m})
                sensitivity_stats.append({"prior":prior,"beta":beta,"alpha":alpha,"rho":rho,
                    **statistics(key,gates,var,protocol,keep)})
            print("Sensitivity",key,fit["success"],flush=True)
    write("gate_sensitivity_metrics.csv",sensitivity)
    write("gate_sensitivity_statistics.csv",sensitivity_stats)
    dump("gate_sensitivity_fits.json",sensitivity_fits)
    # Three fixed clean-pool query-only deletion realizations; no model refitting.
    tq=q[te];tc=c[te];ne=neighbors[tc];unique_q=np.unique(tq)
    query_lookup={doc:i for i,doc in enumerate(unique_q)};query_order=np.array([query_lookup[doc] for doc in tq])
    cn=x[te,:,2]*scale[2]+center[2]
    deletion=[]
    for seed in [17,29,43]:
        rng=np.random.default_rng(seed);records=[]
        for i in unique_q:
            current=dict(patents[i])
            for field in ["title","claims"]:current[field]=old.drop_chars(patents[i][field],.5,rng)
            current["abstract"]="prefix。"+old.drop_chars(old.clean_abstract(patents[i]["abstract"]),.5,rng)
            records.append(current)
        nv=[v.transform(texts) for v,texts in zip(model["vectorizers"],old.Representation.fields(records))]
        nx=sparse.hstack(nv,format="csr")/np.sqrt(3)
        npred=model["topic_classifier"].predict_proba(nx)
        un=old.unions(npred,model["ancestor_matrices"])
        ll=np.mean([old.rowsim(a[query_order],b[tc]) for a,b in zip(nv,views)],axis=0)
        hh=old.hscore([a[query_order] for a in un],[a[tc] for a in u])
        lnn=np.column_stack([np.mean([old.rowsim(a[query_order],b[ne[:,j]]) for a,b in zip(nv,views)],axis=0) for j in range(3)])
        hnn=np.column_stack([old.hscore([a[query_order] for a in un],[a[ne[:,j]] for a in u]) for j in range(3)])
        raw=np.stack([np.ones_like(lnn),lnn,cn,abs(hnn-hh[:,None]),np.repeat(ent[tc,None],3,axis=1),ent[ne]],axis=2)
        xx=(raw-center)/scale
        scores={}
        for objective in ["expected","mean"]:
            chosen=selections[objective];a=chosen["alpha"]
            scores[objective]=evaluate(xx,(1-a)*ll+a*hh,(1-a)*lnn+a*hnn,chosen["theta"],chosen["rho"])[0]
        scores["tuned_no_graph"]=(1-no_graph["alpha"])*ll+no_graph["alpha"]*hh
        nz=model["svd"].transform(nx)[query_order]
        pair_x=np.column_stack([nz,z[tc],abs(nz-z[tc]),nz*z[tc]])
        scores["MLP"]=np.mean([np.clip(m.predict(model["scaler"].transform(pair_x)),0,1) for m in model["mlps"]],axis=0)
        scores["TF-IDF"]=ll
        for method,score in scores.items():
            m,_=old.summarize(score,testpairs,clean,bootstrap=False)
            if method=="TF-IDF":
                cm,_=old.summarize(model["calibration"]["TF-IDF"].predict(score),testpairs,clean,bootstrap=False)
                m["mae"],m["rmse"]=cm["mae"],cm["rmse"]
            deletion.append({"method":method,"protocol":"text_disjoint","rate":.5,"seed":seed,**m})
        print("Clean 50% deletion seed",seed,flush=True)
    write("gate_clean_deletion.csv",deletion)
    deletion_aggregate=[]
    for method in ["expected","mean","tuned_no_graph","MLP","TF-IDF"]:
        rows=[r for r in deletion if r["method"]==method]
        deletion_aggregate.append({"method":method,"protocol":"text_disjoint","rate":.5,"seeds":3,
            **{f"{metric}_{stat}":float(func([r[metric] for r in rows]))
               for metric in ["ndcg10","mae"] for stat,func in [("mean",np.mean),("min",np.min),("max",np.max)]}})
    write("gate_clean_deletion_summary.csv",deletion_aggregate)
    ending_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    assert ending_hashes==source_hashes,"Old source files changed during gate diagnostics"
    summary={"scope":"Old supplied family split diagnostic, not new strict split","grid":{"alpha":alphas,"rho":rhos},
        "prior":.25,"beta":.002,"objectives":["expected","mean","listnet"],"listnet_temperature":TEMPERATURE,
        "selection":"Each objective refitted for every alpha/rho cell; validation nDCG10 then lower calibrated-scale MAE",
        "selected":selections,"tuned_uniform":uniform,"tuned_no_graph":no_graph,
        "representation_reconstruction_seconds":representation_seconds,"used_feature_cache":cached,
        "reconstruction":{"frozen_hber_max_abs_error":old_match,"center_max_abs_error":center_error,"scale_max_abs_error":scale_error},
        "timing_seconds":timings,"total_seconds":time.perf_counter()-start,"maxiter":maxiter,
        "sensitivity":"Selected mean alpha/rho fixed; beta/prior altered with refit, no sensitivity selection on test",
        "source_sha256":source_hashes,"input_hashes_unchanged":True,"bootstrap":"2000 query-family cluster resamples",
        "clean_deletion":"50% query-only character deletion; seeds17/29/43; fixed original focal-clean pools; candidates/neighbors frozen"}
    dump("gate_old_split_summary.json",summary)
    make_report(root,out,summary,metric_rows,stats,bootstrap,sensitivity,deletion_aggregate)
    print("Completed old-split gate diagnostics",round(summary["total_seconds"],2),"seconds",flush=True)


def make_report(root,out,summary,metrics,statistics,bootstrap,sensitivity,deletion):
    """Write measured findings without editing the manuscript or previous results."""
    import pandas as pd
    import json
    m=pd.DataFrame(metrics);st=pd.DataFrame(statistics);bs=pd.DataFrame(bootstrap);sen=pd.DataFrame(sensitivity)
    def table(frame,columns):
        selected=frame[columns].copy()
        for col in selected.columns:
            if selected[col].dtype.kind in "fc":selected[col]=selected[col].map(lambda v:f"{v:.6f}")
        return "| "+" | ".join(columns)+" |\n|"+"|".join(["---"]*len(columns))+"|\n"+"\n".join("| "+" | ".join(map(str,row))+" |" for row in selected.itertuples(index=False,name=None))
    lines=["# 原有划分的门控机理诊断", "", "本报告仅诊断旧的给定family划分；新严格划分由另一独立流程处理。原始数据、已有实验结果、正文和图均未修改。所有结果实际运行，未预设门控优于确定性融合。", "",
        "## 目标、数值实现与模型选择", "",
        "比较三种目标：expected为子图MSE的精确期望；mean为精确期望分数的MSE；listnet为按query平均的ListNet交叉熵，标签分布取softmax(3y)，预测分布取softmax(mean/0.1)。三者均加同一Bernoulli KL，默认prior=0.25、beta=0.002，只对允许边取平均。禁止边概率严格为0。稳定sigmoid及logaddexp避免logit截断造成的梯度不一致。", "",
        "三种目标分别在alpha=[0,.25,.5,.75,1]、rho=[0,.15,.3,.5]的同一20格网格逐格重新训练；以验证集nDCG10最大、MAE最小打破精确平局进行选择。每次L-BFGS-B最多150次迭代。确定均匀及无图对照也独立使用相同验证网格。", "",
        "已有训练向量器/SVD及完整训练集主题分类器来自原模型包；训练节点的主题概率按原三折family分组重新得到OOF预测。复原的原HBER预测与冻结CSV最大绝对误差为 "+f"{summary['reconstruction']['frozen_hber_max_abs_error']:.3g}"+"。", "",
        "六项有限差分（包括禁止边）通过，最大绝对误差小于1e-7；另检验极端logit=1000、全禁边与期望MSE分解。逐项数值见gate_gradient_checks.csv。", "", "选定参数：", ""]
    for obj,record in summary["selected"].items():
        lines.append(f"- {obj}：alpha={record['alpha']}，rho={record['rho']}，validation nDCG10={record['ndcg10']:.6f}，success={record['success']}；theta={np.array2string(record['theta'],precision=6,separator=', ')}。")
    lines += ["", "## 原始与焦点文本去重候选池", "",
        "焦点文本去重仍沿用旧协议：删除query/candidate与train/validation完全同文的pair，保留至少5候选的query。其保留集合在三目标和所有对照间保持一致。TF-IDF排序用原始分数，MAE用原有训练isotonic映射。", "",
        table(m,["method","protocol","queries","pairs","ndcg10","mae"]),"",
        "## 门控分布与掩码方差", "",
        "统计按对应协议实际保留pair计算。saturation_below_001指p<0.01；saturation_above_099指p>0.99。完整1/5/25/50/75/95/99百分位数、极值、p<.001和p>.999比率存于gate_objective_statistics.csv。expected_edges为每pair三条概率之和，mask_variance为八种mask分数的精确加权方差。", "",
        table(st,["method","protocol","gate_mean","gate_q05","gate_q50","gate_q95","saturation_below_001","saturation_above_099","expected_edges_mean","mask_variance_mean"]),"",
        "## Query-family聚类bootstrap", "",
        "先求每query的nDCG10差值，再以query专利family为聚类单位有放回重采样2,000次。若family含多个query，整簇共同进入样本；每次用query差值和除以query数保持query宏平均口径。区间为2.5%与97.5%分位数，未作多重比较校正。区间跨0时，数据未建立该目标与比较方法间的有方向差异。", "",
        table(bs,["objective","protocol","comparator","query_family_clusters","ndcg10_difference","low","high"]),"",
        "## 均值目标固定系数敏感性", "",
        "仅对选定mean目标固定alpha/rho后改变beta=[0,.0002,.002,.02]和prior=[.1,.25,.5]，重新拟合theta。这里没有按测试表现重选超参数；未正则beta=0时不同prior只改变初始intercept。所有theta与optimizer状态见gate_sensitivity_fits.json，分布统计见gate_sensitivity_statistics.csv。", "",
        table(sen,["prior","beta","alpha","rho","protocol","success","ndcg10","mae"]),"",
        "## 焦点文本去重池50%查询删除", "",
        "用种子17、29、43独立删除去除主题前缀后的查询字符，候选/邻居表示与所有参数固定；不重拟合、不按扰动结果重新选参。均值和范围均跨这三个扰动实例，范围不代表置信区间。", "",
        table(pd.DataFrame(deletion),["method","ndcg10_mean","ndcg10_min","ndcg10_max","mae_mean"]),"",
        "## 解释边界与结果文件", "",
        "expected目标等于mean目标的数据项加mask分数方差，因此两者比较检验的是额外结构方差惩罚。期望输出仍是基础分数的确定凸组合，模型内部mask方差不能直接解释为已校准的参数不确定性。原有语义链接来源与主题生成规则的潜在同源关系不由此诊断消除；所有标签仍是主题层级技术邻近目标。", "",
        "主要文件均为results/revision/gate_*：validation_grid、model_fits、objective_metrics、query_metrics、pair_predictions、objective_statistics、family_bootstrap、sensitivity_metrics/statistics/fits、clean_deletion及summary。完整输入SHA256、目标/网格/重建误差与耗时见gate_old_split_summary.json；运行前后输入哈希一致。", ""]
    (root/"notes/gate_revision_report.md").write_text("\n".join(lines))


def augment_old_split():
    """Separate fixed-coefficient loss effects from joint model selection."""
    import json
    import pickle
    import hashlib
    from pathlib import Path
    import pandas as pd
    import run_experiments as old

    root=Path(__file__).resolve().parents[1];out=root/"results/revision"
    summary=json.loads((out/"gate_old_split_summary.json").read_text())
    for path,digest in summary["source_sha256"].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
    cache=np.load(out/"gate_old_split_cache.npz")
    x,lex,hier,ln,hn=[cache[k] for k in ["x","lex","hier","ln","hn"]]
    pairs=old.read("retrieval_pairs");patents=old.read("patents")
    te=np.array([r["split"]=="test" for r in pairs]);testpairs=[r for r in pairs if r["split"]=="test"]
    frozen=pd.read_csv(root/"results/pair_predictions.csv")
    saved=pd.read_csv(out/"gate_pair_predictions.csv")
    clean=saved.focal_clean_retained.to_numpy(dtype=bool)
    families={r["patent_id"]:r["family_id"] for r in patents}
    with (root/"results/fitted_models.pkl").open("rb") as f:model=pickle.load(f)
    fits=json.loads((out/"gate_model_fits.json").read_text())
    fixed={};stats=[];metrics=[];query_rows=[];score_columns={};all_query={}
    original_base=.5*lex+.5*hier;original_neighbors=.5*ln+.5*hn
    def gate_stats(name,p,v,protocol,keep):
        gp=p[keep].ravel();variance=v[keep];expected_edges=p[keep].sum(axis=1)
        return {"method":name,"protocol":protocol,"pairs":int(keep.sum()),
            "gate_mean":float(gp.mean()),"gate_q05":float(np.quantile(gp,.05)),
            "gate_q50":float(np.quantile(gp,.5)),"gate_q95":float(np.quantile(gp,.95)),
            "saturation_below_001":float((gp<.01).mean()),"saturation_above_099":float((gp>.99).mean()),
            "expected_edges_mean":float(expected_edges.mean()),"mask_variance_mean":float(variance.mean()),
            "mask_variance_max":float(variance.max())}
    variants={"old_frozen_hber":frozen.loc[te,"HBER"].to_numpy(),
              "fixed_no_graph":original_base[te],
              "fixed_uniform":(.7*original_base+.3*(original_base+original_neighbors.sum(axis=1))/4)[te],
              "tuned_no_graph":saved.tuned_no_graph.to_numpy(),"tuned_uniform":saved.tuned_uniform.to_numpy()}
    for objective in ["expected","mean","listnet"]:
        fit=next(f for f in fits if f["objective"]==objective and f["alpha"]==.5 and f["rho"]==.3)
        name="fixed_"+objective
        score,p,v=evaluate(x[te],original_base[te],original_neighbors[te],fit["theta"],.3)
        variants[name]=score;fixed[objective]=fit
        for protocol,keep in [("original",np.ones(len(score),dtype=bool)),("text_disjoint",clean)]:
            stats.append(gate_stats(name,p,v,protocol,keep))
    old_score,old_p,old_var=evaluate(x[te],original_base[te],original_neighbors[te],model["theta"],.3)
    assert np.max(abs(old_score-variants["old_frozen_hber"]))<1e-7
    for protocol,keep in [("original",np.ones(len(old_score),dtype=bool)),("text_disjoint",clean)]:
        stats.append(gate_stats("old_frozen_hber",old_p,old_var,protocol,keep))
    for method,score in variants.items():
        score_columns[method]=score
        for protocol,keep in [("original",None),("text_disjoint",clean)]:
            met,pq=old.summarize(score,testpairs,keep,bootstrap=False)
            metrics.append({"method":method,"protocol":protocol,"alpha":.5 if method.startswith("fixed_") else np.nan,
                            "rho":0 if method=="fixed_no_graph" else .3 if method.startswith("fixed_") else np.nan,**met})
            all_query[(method,protocol)]=pq
            query_rows.extend({"method":method,"protocol":protocol,"family":families[r["query"]],**r} for r in pq)
    existing=pd.read_csv(out/"gate_query_metrics.csv")
    for (method,protocol),frame in existing.groupby(["method","protocol"]):
        all_query[(method,protocol)]=frame.to_dict("records")

    def contrast(method,comparator,protocol):
        a={r["query"]:r["ndcg10"] for r in all_query[(method,protocol)]}
        b={r["query"]:r["ndcg10"] for r in all_query[(comparator,protocol)]}
        queries=sorted(a.keys()&b.keys())
        diff=np.array([a[q]-b[q] for q in queries]);fam,inv=np.unique([families[q] for q in queries],return_inverse=True)
        sums=np.bincount(inv,weights=diff);counts=np.bincount(inv)
        rng=np.random.default_rng(20261030);sample=rng.integers(0,len(fam),(2000,len(fam)))
        boot=sums[sample].sum(axis=1)/counts[sample].sum(axis=1)
        return {"method":method,"comparator":comparator,"protocol":protocol,"queries":len(queries),
            "query_family_clusters":len(fam),"ndcg10_difference":float(diff.mean()),
            "low":float(np.quantile(boot,.025)),"high":float(np.quantile(boot,.975)),
            "identical_query_rank_metric":bool(np.all(diff==0)),"replicates":2000}

    contrasts=[]
    for protocol in ["original","text_disjoint"]:
        for method in ["old_frozen_hber","fixed_expected","fixed_mean","fixed_listnet"]:
            for comparator in ["tuned_no_graph","tuned_uniform","fixed_no_graph","fixed_uniform"]:
                contrasts.append(contrast(method,comparator,protocol))
        for a,b in [("mean","expected"),("fixed_mean","fixed_expected"),("fixed_listnet","fixed_expected"),
                    ("fixed_mean","old_frozen_hber")]:
            contrasts.append(contrast(a,b,protocol))
    pd.DataFrame(metrics).to_csv(out/"gate_fixed_objective_metrics.csv",index=False)
    pd.DataFrame(stats).to_csv(out/"gate_fixed_objective_statistics.csv",index=False)
    pd.DataFrame(query_rows).to_csv(out/"gate_fixed_query_metrics.csv",index=False)
    pd.DataFrame(contrasts).to_csv(out/"gate_direct_family_bootstrap.csv",index=False)
    pred=saved[["query","candidate","family","grade","focal_clean_retained"]].copy()
    for method,score in score_columns.items():pred[method]=score
    pred.to_csv(out/"gate_fixed_pair_predictions.csv",index=False)
    (out/"gate_fixed_objective_fits.json").write_text(json.dumps(fixed,indent=2)+"\n")
    # Measure the loss decomposition on the actual OOF/held-out representations.
    risk=[]
    labels=np.array([int(r["oracle_relevance"])/3 for r in pairs])
    split=np.array([r["split"] for r in pairs])
    clean_full=np.zeros(len(pairs),dtype=bool);clean_full[te]=clean
    models={**{objective:{**record} for objective,record in summary["selected"].items()},
            "old_frozen_hber":{"alpha":.5,"rho":.3,"theta":model["theta"]}}
    for name,record in models.items():
        alpha,rho=record["alpha"],record["rho"]
        base=(1-alpha)*lex+alpha*hier;bn=(1-alpha)*ln+alpha*hn
        mean,p,var=evaluate(x,base,bn,record["theta"],rho)
        _,_,weights=_distribution(x,np.asarray(record["theta"]),np.ones((len(x),3),dtype=bool))
        values=_values(base,bn,rho)
        expected_error=np.sum(weights*(values-labels[:,None])**2,axis=1)
        for protocol,keep in [("train_oof",split=="train"),("validation",split=="validation"),
                              ("original",te),("text_disjoint",clean_full)]:
            mean_error=float(np.mean((mean[keep]-labels[keep])**2))
            mean_variance=float(var[keep].mean());expected_mse=float(expected_error[keep].mean())
            risk.append({"method":name,"protocol":protocol,"pairs":int(keep.sum()),
                         "mean_score_mse":mean_error,"mean_mask_variance":mean_variance,
                         "expected_mask_mse":expected_mse,"decomposition_abs_error":abs(expected_mse-mean_error-mean_variance)})
    pd.DataFrame(risk).to_csv(out/"gate_risk_decomposition.csv",index=False)
    summary["comparison_design_caveat"]=("Old manuscript fitted each alpha only at training rho=.3 then validated four inference rhos; "
        "this diagnostic refits each alpha/rho grid cell for all three objectives. Jointly selected rows reflect both loss and refit design. "
        "The fixed alpha=.5/rho=.3 contrast isolates the loss under equal representations and optimization settings.")
    summary["fixed_objective_fits"]=fixed
    summary["frozen_old_hber_included"]=True
    summary["direct_family_paired_contrasts"]=len(contrasts)
    (out/"gate_old_split_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    # Preserve the original measured report and append explicitly separated controls.
    def table(frame,cols):
        rows=[]
        for values in frame[cols].itertuples(index=False,name=None):
            rows.append("| "+" | ".join(f"{v:.6f}" if isinstance(v,(float,np.floating)) else str(v) for v in values)+" |")
        return "| "+" | ".join(cols)+" |\n|"+"|".join(["---"]*len(cols))+"|\n"+"\n".join(rows)
    exact=pd.DataFrame(contrasts)
    useful=exact[(exact.method.isin(["old_frozen_hber","fixed_mean","fixed_listnet"])) &
                 exact.comparator.isin(["tuned_uniform","tuned_no_graph","fixed_expected"]) | 
                 (exact.method.eq("mean")&exact.comparator.eq("expected"))]
    stat=pd.DataFrame(stats)
    sensitivity_stat=pd.read_csv(out/"gate_sensitivity_statistics.csv")
    soft=sensitivity_stat[sensitivity_stat.protocol.eq("original") & sensitivity_stat.beta.eq(.02) & sensitivity_stat.prior.eq(.25)].iloc[0]
    interpretation=["", "## 固定原系数与冻结初稿模型：隔离目标效应", "",
        "初稿的调参方式是每个alpha仅在训练rho=.3拟合，再比较四个推断rho。本轮三目标都在每个(alpha,rho)格重新拟合，因此联合选参行同时改变了目标和拟合设计，不能直接全部归因于损失函数。为隔离目标效应，这里额外取固定alpha=.5、rho=.3的三目标模型，不再重调，并加入冻结的初稿HBER原分数。固定模型复用本轮网格中同一格的真实训练结果；没有额外选择测试最优项。", "",
        table(pd.DataFrame(metrics),["method","protocol","ndcg10","mae"]),"",
        "下表为直接配对的family cluster bootstrap差异（method减comparator）。完整40项含固定/独立调参的no graph与uniform见gate_direct_family_bootstrap.csv；mean减expected单独给出，避免仅凭各自均值判断纯目标差异。", "",
        table(useful,["method","comparator","protocol","ndcg10_difference","low","high"]),"",
        "冻结初稿模型与固定目标模型的门控分布：", "",
        table(stat,["method","protocol","gate_mean","saturation_below_001","saturation_above_099","expected_edges_mean","mask_variance_mean"]),"",
        "本轮稳定梯度无logit clipping，门控系数可达到数千至数万，部分有效概率因此数值上接近0或1。Bernoulli KL在边界取有限值，且logit梯度随sigmoid导数衰减，故默认正则并不保证训练后保持非退化结构分布。必须结合实际饱和率和近零mask分数方差解释结果；不能把名义上的随机门控自动称作已校准、仍然丰富的后验不确定性。", "",
        f"敏感性中将beta从0.002增至0.02、prior固定0.25后，原始测试平均gate为{soft.gate_mean:.6f}，p<.01或p>.99的合计比率为{soft.saturation_below_001+soft.saturation_above_099:.6f}，平均mask分数方差为{soft.mask_variance_mean:.6g}。较强KL可得到更软的结构分布，但这些敏感性设置没有按测试性能重新选参。", "",
        "gate_risk_decomposition.csv直接保存每个选定目标及冻结初稿模型在train OOF、validation、原始测试和焦点清洁测试上的mean-score MSE、精确mask MSE与mask方差，核验expected MSE等于mean MSE加variance。默认新门控近乎确定时，这项额外方差惩罚在最终预测上很小；仅凭目标名称差异不能推定已产生显著不确定性机制。", "",
        "旧划分的未扰动原始池和焦点清洁池上，expected与mean的逐query nDCG10相同；二者保留了略有不同的连续分数和扰动响应。ListNet联合选参后与独立调参uniform的逐query nDCG10也相同。上述结果不支持在这些候选池中把随机图门控作为主要性能来源。50%查询字符删除下mean/expected的差异只由三个固定扰动实例描述，尚未建立新的显著性结论。", ""]
    report=root/"notes/gate_revision_report.md"
    text=report.read_text().split("\n## 固定原系数与冻结初稿模型：隔离目标效应")[0]
    report.write_text(text+"\n"+"\n".join(interpretation))
    for path,digest in summary["source_sha256"].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
    print("Completed fixed-coefficient and direct paired controls",len(contrasts),flush=True)


if __name__ == "__main__":
    import argparse
    import json
    from threadpoolctl import threadpool_limits
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-old-split",action="store_true")
    parser.add_argument("--augment-old-split",action="store_true")
    parser.add_argument("--maxiter",type=int,default=150)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.run_old_split:
            run_old_split(args.maxiter)
            augment_old_split()
        elif args.augment_old_split:
            augment_old_split()
        else:
            print(json.dumps(check_gradients(),indent=2))
