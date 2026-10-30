"""Reproduce graded reranking experiments without using held-out topic labels as inputs."""
from pathlib import Path
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import csv
import hashlib
import itertools
import json
import platform
import pickle
import time
import warnings
from collections import Counter, defaultdict
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
from scipy import sparse
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEEDS = [17, 29, 43]
MASKS = np.array(list(itertools.product([0., 1.], repeat=3)))
PRIOR, BETA = .25, .002
METHODS = ["BM25", "TF-IDF", "MLP", "Jaccard", "HBER"]


def read(name):
    with (ROOT / "data" / (name + ".csv")).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, data):
    data = list(data)
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def fulltext(r):
    return tuple(r[k] for k in ["title", "abstract", "claims", "keywords_json"])


def clean_abstract(t):
    return t.split("。", 1)[1] if "。" in t else ""


def drop_chars(t, rate, rng):
    if not rate:
        return t
    return "".join(c for c, retain in zip(t, rng.random(len(t)) >= rate) if retain)


def ancestor_matrices(topics, leaves):
    lookup = {r["topic_id"]: r for r in topics}
    mats = []
    for level in [1, 2, 3]:
        nodes = [r["topic_id"] for r in topics if int(r["level"]) == level]
        a = np.zeros((len(leaves), len(nodes)))
        for i, leaf in enumerate(leaves):
            t = leaf
            while int(lookup[t]["level"]) > level:
                t = lookup[t]["parent_topic_id"]
            a[i, nodes.index(t)] = 1
        mats.append(a)
    return mats


def unions(p, mats):
    log_absence = np.log1p(-np.clip(p, 1e-7, 1-1e-7))
    return [-np.expm1(log_absence @ a) for a in mats]


def entropy(p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return -(p*np.log(p)+(1-p)*np.log1p(-p)).mean(axis=1) / np.log(2)


def hscore(ua, ub):
    return np.mean([1-np.prod(1-a*b, axis=1) for a, b in zip(ua, ub)], axis=0)


def rowsim(x, y):
    return np.asarray(x.multiply(y).sum(axis=1)).ravel()


class Representation:
    def fit(self, records, train, y):
        begin = time.perf_counter()
        self.vec = []
        fields = self.fields(records)
        views = []
        for texts in fields:
            v = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2,
                                max_features=12000, sublinear_tf=True, dtype=np.float64)
            v.fit([texts[i] for i in train])
            views.append(v.transform(texts))
            self.vec.append(v)
        self.x = sparse.hstack(views, format="csr") / np.sqrt(3)
        self.svd = TruncatedSVD(n_components=64, random_state=17)
        self.svd.fit(self.x[train])
        self.z = self.svd.transform(self.x)
        self.clf = OneVsRestClassifier(LogisticRegression(C=2., solver="liblinear", max_iter=250))
        self.clf.fit(self.x[train], y)
        self.p = self.clf.predict_proba(self.x)
        # Cross-fitted predictions train the second-stage graph gate.
        groups = [records[i]["family_id"] for i in train]
        for fit_idx, hold_idx in GroupKFold(n_splits=3).split(train, y, groups):
            inner = OneVsRestClassifier(LogisticRegression(C=2., solver="liblinear", max_iter=250))
            inner.fit(self.x[train[fit_idx]], y[fit_idx])
            self.p[train[hold_idx]] = inner.predict_proba(self.x[train[hold_idx]])
        self.fit_seconds = time.perf_counter() - begin
        self.views = views
        return self

    @staticmethod
    def fields(records):
        return [[r["title"] for r in records],
                [clean_abstract(r["abstract"]) for r in records],
                [r["claims"] for r in records]]

    def transform(self, records):
        views = [v.transform(t) for v, t in zip(self.vec, self.fields(records))]
        x = sparse.hstack(views, format="csr") / np.sqrt(3)
        return {"views": views, "z": self.svd.transform(x), "p": self.clf.predict_proba(x)}


def pair_features(z, q, c):
    a, b = z[q], z[c]
    return np.column_stack([a, b, abs(a-b), a*b])


def summarize(scores, pairs, keep=None, bootstrap=True):
    groups = defaultdict(list)
    for i, r in enumerate(pairs):
        if keep is None or keep[i]:
            groups[r["query_patent_id"]].append(i)
    per_query = []
    for qid, idx in groups.items():
        if keep is not None and len(idx) < 5:
            continue
        idx = np.array(idx)
        y = np.array([int(pairs[i]["oracle_relevance"]) for i in idx])
        s = scores[idx]
        # Stable identifier tie breaking, independent of CSV input order and labels.
        cids = np.array([pairs[i]["candidate_patent_id"] for i in idx])
        order = np.lexsort((cids, -s))
        ideal = np.sort(y)[::-1]
        row = {"query": qid, "candidates": len(idx)}
        for k in [1, 3, 5, 10]:
            kk = min(k, len(idx)); discount = np.log2(np.arange(2, kk+2))
            dcg = np.sum((2.**y[order[:kk]]-1)/discount)
            idcg = np.sum((2.**ideal[:kk]-1)/discount)
            row[f"ndcg{k}"] = float(dcg/idcg) if idcg else np.nan
        relevant = y >= 2
        found = np.flatnonzero(relevant[order])
        row["mrr"] = float(1/(found[0]+1)) if len(found) else 0.
        row["precision3"] = float(relevant[order[:min(3, len(idx))]].mean())
        row["mae"] = float(np.abs(3*s-y).mean())
        row["rmse_squared"] = float(((3*s-y)**2).mean())
        per_query.append(row)
    assert per_query
    result = {"queries": len(per_query), "pairs": sum(r["candidates"] for r in per_query),
              "min_candidates": min(r["candidates"] for r in per_query),
              "max_candidates": max(r["candidates"] for r in per_query)}
    for k in ["ndcg1", "ndcg3", "ndcg5", "ndcg10", "mrr", "precision3", "mae"]:
        values = np.array([r[k] for r in per_query]); values = values[np.isfinite(values)]
        result[k] = float(values.mean())
        if bootstrap and k in ["ndcg10", "mae"]:
            rng = np.random.default_rng(20261030)
            boot = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
            result[k+"_low"], result[k+"_high"] = map(float, np.quantile(boot, [.025, .975]))
    result["rmse"] = float(np.sqrt(np.mean([r["rmse_squared"] for r in per_query])))
    result["queries_with_positive_gain"] = sum(np.isfinite(r["ndcg10"]) for r in per_query)
    return result, per_query


def expectation(x, base, neighbors, theta, rho, with_loss=None, allowed=None):
    logits = x @ theta
    p = expit(np.clip(logits, -25, 25))
    if allowed is not None:
        p = p * allowed
    weights = np.prod(np.where(MASKS[None, :, :] == 1,
                               p[:, None, :], 1-p[:, None, :]), axis=2)
    values = (1-rho)*base[:, None] + rho*(base[:, None]+neighbors @ MASKS.T)/(1+MASKS.sum(axis=1))
    assert np.max(abs(weights.sum(axis=1)-1)) < 1e-10
    if with_loss is None:
        return np.sum(weights*values, axis=1), p
    error = (values-with_loss[:, None])**2
    loss = np.mean(np.sum(weights*error, axis=1))
    kl = p*np.log(p/PRIOR)+(1-p)*np.log((1-p)/(1-PRIOR))
    loss += BETA * kl.mean()
    # Derivative of the product Bernoulli probability through logits.
    d = np.einsum("nm,nme,ned->d", weights*error,
                  MASKS[None, :, :]-p[:, None, :], x) / len(x)
    d += BETA * np.einsum("ne,ned->d",
        p*(1-p)*np.log(p*(1-PRIOR)/(PRIOR*(1-p))), x) / p.size
    return float(loss), d


def main():
    start_all = time.perf_counter()
    initial_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"data").glob("*.csv")}
    patents, pairs, edges, assignments, topics = [read(n) for n in
        ["patents", "retrieval_pairs", "graph_edges", "oracle_patent_topics", "oracle_topic"]]
    assert initial_hashes == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"data").glob("*.csv")}
    ids = {r["patent_id"]: i for i, r in enumerate(patents)}
    train = np.array([i for i,r in enumerate(patents) if r["split"]=="train"])
    leaves = [r["topic_id"] for r in topics if r["level"]=="3"]
    assign = {r["patent_id"]: r for r in assignments}
    # Only training labels form the supervised target matrix.
    yleaf = np.zeros((len(train), len(leaves)), dtype=int)
    for j,i in enumerate(train):
        for key in ["primary_topic_id", "secondary_topic_id"]:
            t = assign[patents[i]["patent_id"]][key]
            if t: yleaf[j, leaves.index(t)] = 1
    rep = Representation().fit(patents, train, yleaf)
    print("Representation fitted", round(rep.fit_seconds, 2), flush=True)
    mats = ancestor_matrices(topics, leaves)
    u = unions(rep.p, mats); ent = entropy(rep.p)
    ns = defaultdict(list)
    for r in edges: ns[ids[r["source_patent_id"]]].append(ids[r["target_patent_id"]])
    neighbors = np.array([ns[i] for i in range(len(patents))])
    assert neighbors.shape==(len(patents),3)
    q = np.array([ids[r["query_patent_id"]] for r in pairs])
    c = np.array([ids[r["candidate_patent_id"]] for r in pairs])
    y = np.array([int(r["oracle_relevance"])/3 for r in pairs])
    split = np.array([r["split"] for r in pairs]); tr=split=="train"; va=split=="validation"; te=split=="test"
    lexical = np.mean([rowsim(v[q],v[c]) for v in rep.views],axis=0)
    def features(query, candidate, neigh, current=None):
        v = rep.views if current is None else current["views"]
        p = rep.p if current is None else current["p"]
        uu = u if current is None else unions(p,mats)
        ee = ent if current is None else entropy(p)
        lex=np.mean([rowsim(t[query],t[candidate]) for t in v],axis=0)
        hier=hscore([t[query] for t in uu],[t[candidate] for t in uu])
        ln=np.column_stack([np.mean([rowsim(t[query],t[neigh[:,j]]) for t in v],axis=0) for j in range(3)])
        hn=np.column_stack([hscore([t[query] for t in uu],[t[neigh[:,j]] for t in uu]) for j in range(3)])
        cn=np.column_stack([np.mean([rowsim(t[candidate],t[neigh[:,j]]) for t in v],axis=0) for j in range(3)])
        xx=np.stack([np.ones_like(ln),ln,cn,abs(hn-hier[:,None]),np.repeat(ee[candidate,None],3,axis=1),ee[neigh]],axis=2)
        return lex,hier,ln,hn,xx
    lex,hier,ln,hn,xx=features(q,c,neighbors[c])
    center=xx[tr].reshape(-1,6).mean(axis=0); scale=xx[tr].reshape(-1,6).std(axis=0)
    center[0]=0;scale[0]=1;scale[scale<1e-6]=1
    xx=(xx-center)/scale
    validation=[]; fits={}
    for alpha in [0.,.25,.5,.75,1.]:
        base=(1-alpha)*lex+alpha*hier; bn=(1-alpha)*ln+alpha*hn
        theta0=np.zeros(6);theta0[0]=np.log(PRIOR/(1-PRIOR))
        fit=minimize(lambda th:expectation(xx[tr],base[tr],bn[tr],th,.3,y[tr]),theta0,jac=True,
                     method="L-BFGS-B",options={"maxiter":100,"ftol":1e-10})
        fits[alpha]=fit.x
        for rho in [0.,.15,.3,.5]:
            score,_=expectation(xx[va],base[va],bn[va],fit.x,rho)
            m,_=summarize(score,[r for r,s in zip(pairs,va) if s],bootstrap=False)
            validation.append({"alpha":alpha,"rho":rho,"ndcg10":m["ndcg10"],"mae":m["mae"],
                               "optimizer_success":bool(fit.success),"train_objective":float(fit.fun)})
        print("Gate",alpha,fit.success,round(fit.fun,5),flush=True)
    # Reranking is primary; MAE breaks exact ranking ties.
    best=max(validation,key=lambda r:(r["ndcg10"],-r["mae"]))
    alpha,rho=best["alpha"],best["rho"];theta=fits[alpha]
    base=(1-alpha)*lex+alpha*hier;bn=(1-alpha)*ln+alpha*hn
    hber,gates=expectation(xx,base,bn,theta,rho)
    print("Selected",best,flush=True)
    # BM25 uses training document frequency; no test document changes its IDF.
    combined=[" ".join(t) for t in zip(*Representation.fields(patents))]
    count=CountVectorizer(analyzer="char",ngram_range=(2,2),min_df=2,max_features=12000,dtype=np.float64)
    count.fit([combined[i] for i in train]);counts=count.transform(combined).tocsr()
    df=np.asarray((counts[train]>0).sum(axis=0)).ravel()
    idf=np.log(1+(len(train)-df+.5)/(df+.5));lengths=np.asarray(counts.sum(axis=1)).ravel();avg=lengths[train].mean()
    norms=1.2*(.25+.75*lengths/avg)
    bm=counts.copy()
    for i in range(len(patents)):
        sl=slice(bm.indptr[i],bm.indptr[i+1]);f=bm.data[sl]
        bm.data[sl]=f*2.2/(f+norms[i])*idf[bm.indices[sl]]
    query_binary=counts.copy();query_binary.data[:]=1
    bmraw=rowsim(query_binary[q],bm[c])
    calibration={}
    predictions={}
    for name,raw in [("BM25",bmraw),("TF-IDF",lexical)]:
        iso=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1).fit(raw[tr],y[tr])
        calibration[name]=iso
        predictions[name]=iso.predict(raw)
    sets=[set(combined[i]) for i in range(len(patents))]
    jac=np.array([len(sets[a]&sets[b])/max(1,len(sets[a]|sets[b])) for a,b in zip(q,c)])
    calibration["Jaccard"]=IsotonicRegression(out_of_bounds="clip",y_min=0,y_max=1).fit(jac[tr],y[tr])
    predictions["Jaccard"]=calibration["Jaccard"].predict(jac)
    # Rank lexical baselines with raw scores to avoid introducing isotonic ties.
    raw_rank={"BM25":bmraw,"TF-IDF":lexical,"Jaccard":jac}
    xp=pair_features(rep.z,q,c);scaler=StandardScaler().fit(xp[tr]);xp=scaler.transform(xp)
    mlps=[];mlp_predictions=[];training=[]
    for seed in SEEDS:
        model=MLPRegressor(hidden_layer_sizes=(64,32),activation="relu",alpha=.001,
              learning_rate_init=.001,batch_size=256,max_iter=100,early_stopping=True,
              validation_fraction=.1,n_iter_no_change=8,random_state=seed)
        t=time.perf_counter()
        with warnings.catch_warnings(record=True) as ww:
            model.fit(xp[tr],y[tr])
        seconds=time.perf_counter()-t
        pred=np.clip(model.predict(xp),0,1)
        mlps.append(model);mlp_predictions.append(pred)
        training.append({"method":"MLP","seed":seed,"seconds":seconds,"iterations":model.n_iter_,"warnings":[str(w.message) for w in ww]})
        print("MLP",seed,round(seconds,2),model.n_iter_,flush=True)
    predictions["MLP"]=np.mean(mlp_predictions,axis=0);predictions["HBER"]=hber
    testpairs=[r for r,s in zip(pairs,te) if s]
    # Text-disjoint secondary evaluation: remove held-out docs matching train or validation text.
    seen={fulltext(r) for r in patents if r["split"]!="test"}
    clean=np.array([fulltext(patents[a]) not in seen and fulltext(patents[b]) not in seen for a,b in zip(q[te],c[te])])
    metric_rows=[];query_rows=[];metric_map={}
    for name in METHODS:
        for protocol,keep in [("original",None),("text_disjoint",clean)]:
            scored=predictions[name][te]
            m,pq=summarize(scored,testpairs,keep)
            if name in raw_rank:
                rankm,rankpq=summarize(raw_rank[name][te],testpairs,keep)
                for k in list(m):
                    if k.startswith("ndcg") or k in ["mrr","precision3"]:m[k]=rankm[k]
                for a,b in zip(pq,rankpq):
                    for k in list(a):
                        if k.startswith("ndcg") or k in ["mrr","precision3"]:a[k]=b[k]
            metric_map[(name,protocol)]=(m,pq)
            metric_rows.append({"method":name,"protocol":protocol,**m})
            query_rows.extend({"method":name,"protocol":protocol,**r} for r in pq)
    write_csv("metrics.csv",metric_rows);write_csv("query_metrics.csv",query_rows)
    write_csv("validation_grid.csv",validation)
    write_csv("pair_predictions.csv",({"query":r["query_patent_id"],"candidate":r["candidate_patent_id"],"split":r["split"],
         "grade":r["oracle_relevance"],**{name:float(predictions[name][i]) for name in METHODS},
         **{name+"_ranking_score":float(raw_rank[name][i]) for name in raw_rank}} for i,r in enumerate(pairs)))
    # Component removals, plus independently validation-tuned deterministic comparators.
    ablations={"HBER":hber,"No graph":base,"Text only":lex,"Hierarchy only":hier,
               "Uniform neighbors":(1-rho)*base+rho*(base+bn.sum(axis=1))/4,
               "Prior gates":expectation(xx,base,bn,np.array([np.log(PRIOR/(1-PRIOR)),0,0,0,0,0]),rho)[0],
               "No hierarchy":expectation(xx,lex,ln,theta,rho)[0]}
    deterministic_grid=[]
    for aa in [0.,.25,.5,.75,1.]:
        bb=(1-aa)*lex+aa*hier;bnn=(1-aa)*ln+aa*hn
        for rr in [0.,.15,.3,.5]:
            score=(1-rr)*bb+rr*(bb+bnn.sum(axis=1))/4
            m,_=summarize(score[va],[r for r,s in zip(pairs,va) if s],bootstrap=False)
            deterministic_grid.append({"alpha":aa,"rho":rr,"ndcg10":m["ndcg10"],"mae":m["mae"]})
    detbest=max(deterministic_grid,key=lambda r:(r["ndcg10"],-r["mae"]))
    ngbest=max([r for r in deterministic_grid if r["rho"]==0],key=lambda r:(r["ndcg10"],-r["mae"]))
    aa,rr=detbest["alpha"],detbest["rho"];bb=(1-aa)*lex+aa*hier;bnn=(1-aa)*ln+aa*hn
    ablations["Tuned uniform"]=(1-rr)*bb+rr*(bb+bnn.sum(axis=1))/4
    ablations["Tuned no graph"]=(1-ngbest["alpha"])*lex+ngbest["alpha"]*hier
    write_csv("deterministic_validation_grid.csv",deterministic_grid)
    ab_rows=[]
    for name,score in ablations.items():
        for protocol,keep in [("original",None),("text_disjoint",clean)]:
            m,_=summarize(score[te],testpairs,keep)
            ab_rows.append({"variant":name,"protocol":protocol,**m})
    write_csv("ablations.csv",ab_rows)
    # Robustness stress: delete characters in query text only. All candidate/neighborhood text stays fixed.
    tq=q[te];tc=c[te];ne=neighbors[tc]
    stress=[]
    for rate in [0.,.1,.2,.3,.4,.5]:
        for seed in SEEDS:
            rng=np.random.default_rng(seed)
            noisy=[dict(r) for r in patents]
            for i in np.unique(tq):
                for f in ["title","claims"]:noisy[i][f]=drop_chars(patents[i][f],rate,rng)
                noisy[i]["abstract"]="prefix。"+drop_chars(clean_abstract(patents[i]["abstract"]),rate,rng)
            noisyq=rep.transform(noisy)
            # Query corruption only: do not change candidate or neighbor representations even if it also is a query.
            def qlex(a,b):return np.mean([rowsim(v[a],w[b]) for v,w in zip(noisyq["views"],rep.views)],axis=0)
            un=unions(noisyq["p"],mats)
            ll=qlex(tq,tc);hh=hscore([t[tq] for t in un],[t[tc] for t in u])
            lnn=np.column_stack([qlex(tq,ne[:,j]) for j in range(3)])
            hnn=np.column_stack([hscore([t[tq] for t in un],[t[ne[:,j]] for t in u]) for j in range(3)])
            cn=xx[te,:,2]*scale[2]+center[2]
            xf=np.stack([np.ones_like(lnn),lnn,cn,abs(hnn-hh[:,None]),
                          np.repeat(ent[tc,None],3,axis=1),ent[ne]],axis=2)
            xf=(xf-center)/scale
            hscore_noise=expectation(xf,(1-alpha)*ll+alpha*hh,(1-alpha)*lnn+alpha*hnn,theta,rho)[0]
            mlpscore=np.mean([np.clip(m.predict(scaler.transform(np.column_stack([
                noisyq["z"][tq],rep.z[tc],abs(noisyq["z"][tq]-rep.z[tc]),noisyq["z"][tq]*rep.z[tc]]))),0,1) for m in mlps],axis=0)
            for name,score in [("HBER",hscore_noise),("TF-IDF",ll),("MLP",mlpscore),("No graph",(1-alpha)*ll+alpha*hh)]:
                m,_=summarize(score,testpairs,bootstrap=False)
                if name=="TF-IDF":
                    calibrated,_=summarize(calibration["TF-IDF"].predict(score),testpairs,bootstrap=False)
                    m["mae"]=calibrated["mae"]
                stress.append({"stress":"query_character_deletion","rate":rate,"seed":seed,"method":name,"ndcg10":m["ndcg10"],"mae":m["mae"]})
        print("Text stress",rate,flush=True)
    # Rewire within test only; oracle topics never inform replacement choices.
    testids=np.array([i for i,r in enumerate(patents) if r["split"]=="test"])
    for rate in [0.,.25,.5,.75,1.]:
        for seed in SEEDS:
            rng=np.random.default_rng(seed);new=ne.copy();bad=rng.random(new.shape)<rate
            new[bad]=rng.choice(testids,int(bad.sum()))
            ll,hh,lnn,hnn,xf=features(tq,tc,new);xf=(xf-center)/scale
            bb=(1-alpha)*ll+alpha*hh;bnn=(1-alpha)*lnn+alpha*hnn
            expected,p=expectation(xf,bb,bnn,theta,rho)
            uniform=(1-rho)*bb+rho*(bb+bnn.sum(axis=1))/4
            for name,score in [("HBER",expected),("Uniform neighbors",uniform),("No graph",bb)]:
                m,_=summarize(score,testpairs,bootstrap=False)
                stress.append({"stress":"edge_rewiring","rate":rate,"seed":seed,"method":name,"ndcg10":m["ndcg10"],"mae":m["mae"]})
    write_csv("stress_tests.csv",stress)
    # Close every test ego edge whose endpoint has seen train/validation text.
    allowed=np.array([[fulltext(patents[n]) not in seen and fulltext(patents[n])!=fulltext(patents[ci]) for n in nn]
                      for ci,nn in zip(tc,ne)])
    strict_hber=expectation(xx[te],base[te],bn[te],theta,rho,allowed=allowed)[0]
    strict_results=[]
    for name in METHODS:
        score=strict_hber if name=="HBER" else raw_rank.get(name,predictions[name])[te]
        m,_=summarize(score,testpairs,clean)
        if name in raw_rank:
            calibrated,_=summarize(predictions[name][te],testpairs,clean)
            for key in ["mae","mae_low","mae_high","rmse"]:m[key]=calibrated[key]
        strict_results.append({"method":name,"protocol":"ego_text_disjoint",**m})
    write_csv("ego_disjoint_metrics.csv",strict_results)
    # Query domain and year are diagnostic group labels used after predictions are frozen.
    subgroups=[]
    for name in METHODS:
        score=raw_rank.get(name,predictions[name])[te]
        for group in ["domain","year","multilabel"]:
            values=[]
            for r in testpairs:
                a=assign[r["query_patent_id"]]
                values.append(a["primary_topic_id"].split("_")[0] if group=="domain" else
                              patents[ids[r["query_patent_id"]]]["year"] if group=="year" else
                              "multiple" if a["secondary_topic_id"] else "single")
            values=np.array(values)
            for value in sorted(set(values)):
                ix=np.flatnonzero(values==value)
                m,_=summarize(score[ix],[testpairs[i] for i in ix],bootstrap=False)
                subgroups.append({"group":group,"value":value,"method":name,**m})
    write_csv("subgroups.csv",subgroups)
    # Cached document representations; pair similarities and ego features are included.
    latency=[]
    index=np.flatnonzero(te)[:10]
    def hber_timing():
        ll,hh,lnn,hnn,xf=features(q[index],c[index],neighbors[c[index]])
        return expectation((xf-center)/scale,(1-alpha)*ll+alpha*hh,(1-alpha)*lnn+alpha*hnn,theta,rho)[0]
    funcs={"BM25":lambda:rowsim(query_binary[q[index]],bm[c[index]]),
      "TF-IDF":lambda:np.mean([rowsim(v[q[index]],v[c[index]]) for v in rep.views],axis=0),
      "Jaccard":lambda:[len(sets[a]&sets[b])/max(1,len(sets[a]|sets[b])) for a,b in zip(q[index],c[index])],
      "MLP":lambda:np.mean([np.clip(m.predict(scaler.transform(pair_features(rep.z,q[index],c[index]))),0,1) for m in mlps],axis=0),
      "HBER":hber_timing}
    for name,func in funcs.items():
        for _ in range(10):func()
        times=[]
        for _ in range(200):
            t=time.perf_counter_ns();func();times.append((time.perf_counter_ns()-t)/1e6)
        latency.append({"method":name,"median_ms_10_candidates":float(np.median(times)),"p95_ms_10_candidates":float(np.quantile(times,.95)),"scope":"cached document representations; pair features and scoring included; encoding excluded"})
    write_csv("latency.csv",latency)
    # End-to-end encoding and scoring not inferred from cached timings.
    t=time.perf_counter();rep.transform([patents[i] for i in testids]);encoding_seconds=time.perf_counter()-t
    mlp_seed_metrics=[]
    for seed,pred in zip(SEEDS,mlp_predictions):
        m,_=summarize(pred[te],testpairs,bootstrap=False);mlp_seed_metrics.append({"seed":seed,**m})
    # Paired bootstrap against each comparator, with shared query resampling.
    comparisons=[]
    for protocol in ["original","text_disjoint"]:
        hq={r["query"]:r for r in metric_map[("HBER",protocol)][1]}
        for name in METHODS[:-1]:
            cq={r["query"]:r for r in metric_map[(name,protocol)][1]}
            shared=sorted(set(hq)&set(cq));d=np.array([hq[k]["ndcg10"]-cq[k]["ndcg10"] for k in shared])
            d=d[np.isfinite(d)];rng=np.random.default_rng(20261030)
            bs=d[rng.integers(0,len(d),(2000,len(d)))].mean(axis=1)
            comparisons.append({"comparator":name,"protocol":protocol,"queries":len(d),"ndcg10_difference":float(d.mean()),"low":float(np.quantile(bs,.025)),"high":float(np.quantile(bs,.975))})
    write_csv("paired_differences.csv",comparisons)
    objects={"vectorizers":rep.vec,"svd":rep.svd,"topic_classifier":rep.clf,"theta":theta,"center":center,"scale":scale,"alpha":alpha,"rho":rho,"calibration":calibration,"scaler":scaler,"mlps":mlps,"leaves":leaves,"ancestor_matrices":mats}
    with (OUT/"fitted_models.pkl").open("wb") as f:pickle.dump(objects,f,protocol=5)
    state={"method":"Hierarchy-calibrated Bayesian Ego-graph Retrieval","acronym":"HBER",
      "selected":best,"theta":theta.tolist(),"gate_feature_mean":center.tolist(),"gate_feature_scale":scale.tolist(),
      "prior":PRIOR,"beta":BETA,"seeds":SEEDS,"representation_fit_seconds":rep.fit_seconds,
      "tuned_uniform":detbest,"tuned_no_graph":ngbest,"crossfit":"three family-disjoint folds for training topic predictions",
      "training":training,"gate_mean_test":float(gates[te].mean()),"expected_retained_edges_test":float(gates[te].sum(axis=1).mean()),
      "vocabulary_sizes":[len(v.vocabulary_) for v in rep.vec],"model_pickle_bytes":(OUT/"fitted_models.pkl").stat().st_size,
      "test_encoding_seconds":encoding_seconds,"test_encoding_records":len(testids),
      "test_ego_edges_closed_fraction":float(1-allowed.mean()),
      "mlp_seed_metrics":mlp_seed_metrics,"runtime":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"machine":platform.machine(),"processor":platform.processor(),"platform":platform.platform(),"threads":1},
      "elapsed_seconds":time.perf_counter()-start_all,"input_sha256":initial_hashes,
      "protocol":{"primary":"provided family-disjoint ten-candidate pools, prefix removed","secondary":"drop test pairs with query or candidate full text matching any train/validation record; retain queries with at least five candidates","ground_truth":"provided topic-tree grades, not independent patent examination judgments","topic_input":"training labels only; held-out probabilities predicted from text","timing":"ten-pair batch with cached document representations; includes pair and ego features; excludes encoding, measured separately","MLP":"mean clipped predictions across three independent seeds"}}
    for p in (ROOT/"data").glob("*.csv"):
        assert hashlib.sha256(p.read_bytes()).hexdigest()==initial_hashes[p.name],"Input changed"
    for name,pred in predictions.items():assert np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1,(name,pred.min(),pred.max())
    convert=lambda obj:obj.item() if isinstance(obj,np.generic) else str(obj)
    (OUT/"experiment_summary.json").write_text(json.dumps(state,indent=2,default=convert)+"\n")
    print(json.dumps({"selected":best,"metrics":metric_rows,"seconds":state["elapsed_seconds"]},indent=2,default=convert),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):main()
