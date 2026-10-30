"""Controlled topic-relevance ranking under family-and-input-component splits.

This is a new experiment, not a re-use of supplied retrieval pairs.  It only
writes results/revision and never changes source data or the original results.
"""
from __future__ import annotations

import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_name, "1")
import argparse
import csv
import hashlib
import io
import json
import platform
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import xgboost as xgb

from revision_gate import evaluate as evaluate_gate, fit_gate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision"
SEEDS = [17, 29, 43]
ALPHAS = [0., .25, .5, .75, 1.]
RHOS = [0., .15, .3, .5]
PARTITIONS = ["train", "validation", "test"]
QUERY_LIMITS = {"train": 1200, "validation": 250, "test": 350}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=json_default) + "\n")


def write_csv(name, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty result table: " + name)
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stripped_abstract(text):
    return text.split("。", 1)[1] if "。" in text else ""


class DSU:
    def __init__(self, count):
        self.parent = np.arange(count)
        self.size = np.ones(count, dtype=int)

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = int(self.parent[a])
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def audit_data():
    tables, hashes = {}, {}
    for path in sorted((ROOT / "data").glob("*.csv")):
        content = path.read_bytes()
        hashes[path.name] = hashlib.sha256(content).hexdigest()
        tables[path.stem] = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    original = tables["patents"]
    assignments = {r["patent_id"]: r for r in tables["oracle_patent_topics"]}
    dsu = DSU(len(original))
    families, tuples = {}, {}
    text_key = []
    for i, row in enumerate(original):
        key = (row["title"], stripped_abstract(row["abstract"]), row["claims"])
        text_key.append(key)
        if row["family_id"] in families:
            dsu.union(i, families[row["family_id"]])
        else:
            families[row["family_id"]] = i
        if key in tuples:
            dsu.union(i, tuples[key])
        else:
            tuples[key] = i
    members = defaultdict(list)
    for i in range(len(original)):
        members[dsu.find(i)].append(i)
    component_names = {r: "CMP_" + min(original[i]["patent_id"] for i in indices)[4:]
                       for r, indices in members.items()}
    # Lexicographic minimum ID is the fixed representative of each exact input.
    duplicate_groups = defaultdict(list)
    for i, key in enumerate(text_key):
        duplicate_groups[key].append(i)
    representative_of, records = {}, []
    for key, indices in duplicate_groups.items():
        representative = min(indices, key=lambda i: original[i]["patent_id"])
        source = original[representative]
        label_sets = {tuple(assignments[original[i]["patent_id"]][k]
                            for k in ["primary_topic_id", "secondary_topic_id"]) for i in indices}
        if len(label_sets) != 1:
            raise ValueError("Identical model text has conflicting topic labels")
        row = {"patent_id": source["patent_id"], "component_id": component_names[dsu.find(representative)],
               "family_id": source["family_id"], "year": source["year"],
               "title": key[0], "abstract": key[1], "claims": key[2],
               "model_text": " ".join(key), "duplicate_record_count": len(indices)}
        records.append(row)
        for i in indices:
            representative_of[original[i]["patent_id"]] = source["patent_id"]
    records.sort(key=lambda r: r["patent_id"])
    ids = {r["patent_id"]: i for i, r in enumerate(records)}
    component_counts = Counter(r["component_id"] for r in records)
    mapped_edges = set()
    self_edges = 0
    for edge in tables["graph_edges"]:
        a = representative_of[edge["source_patent_id"]]
        b = representative_of[edge["target_patent_id"]]
        if a == b:
            self_edges += 1
        else:
            mapped_edges.add((ids[a], ids[b]))
    audit = {"task": "controlled topic-relevance ranking; ontology-derived labels",
             "input_sha256": hashes, "original_records": len(original),
             "original_families": len(families), "unique_model_inputs": len(records),
             "components": len(members), "split_component_ratios": [.70, .15, .15],
             "largest_components_original": sorted(map(len, members.values()), reverse=True)[:20],
             "largest_components_representatives": component_counts.most_common(20),
             "raw_graph_edges": len(tables["graph_edges"]), "mapped_self_edges_removed": self_edges,
             "unique_mapped_nonself_edges": len(mapped_edges),
             "topic_conflicts_within_identical_inputs": 0,
             "representative_rule": "global smallest patent ID per (title, first-sentence-stripped abstract, claims)",
             "component_rule": "transitive union over same original family OR identical model input",
             "graph_rule": "map all original endpoints to representatives; unique nonself directed edges; then split-local and at most three sorted target IDs",
             "graph_provenance": "unknown; semantic_link is supplied and may encode unobserved construction choices",
             "candidate_rule": "same partition, different DSU component; 20 highest train-vocabulary lexical scores plus 20 random remaining candidates; no label-based sampling",
             "query_rule": "seeded sampling without labels, capped at train1200/validation250/test350",
             "label_rule": "maximum shared ancestor depth across primary/secondary leaf sets; generated only for training supervision and evaluation"}
    write_csv("representatives.csv", records)
    write_csv("representative_mapping.csv", ({"original_id": r["patent_id"],
        "representative_id": representative_of[r["patent_id"]],
        "original_family_id": r["family_id"], "component_id": component_names[dsu.find(i)]}
        for i, r in enumerate(original)))
    write_json("component_audit.json", audit)
    return records, assignments, tables["oracle_topic"], mapped_edges, audit


def split_records(records, seed):
    rng = np.random.default_rng(seed)
    components = np.array(sorted({r["component_id"] for r in records}))
    permutation = rng.permutation(len(components))
    first, second = int(.70 * len(components)), int(.85 * len(components))
    component_partition = {}
    for name, part in zip(PARTITIONS, np.split(permutation, [first, second])):
        component_partition.update({str(components[i]): name for i in part})
    partition = np.array([component_partition[r["component_id"]] for r in records])
    return partition


class Representation:
    def fit_text(self, records, train):
        self.vectorizers, self.views = [], []
        for field in ["title", "abstract", "claims"]:
            texts = [r[field] for r in records]
            vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2,
                                  max_features=12000, sublinear_tf=True, dtype=np.float64)
            vec.fit([texts[i] for i in train])
            self.vectorizers.append(vec)
            self.views.append(vec.transform(texts))
        self.x = sparse.hstack(self.views, format="csr") / np.sqrt(3)
        return self

    def transform(self, records):
        views = [v.transform([r[field] for r in records]) for v, field in
                 zip(self.vectorizers, ["title", "abstract", "claims"])]
        x = sparse.hstack(views, format="csr") / np.sqrt(3)
        return views, x


def topic_targets(records, assignments, topics, primary_only=False):
    leaves = [r["topic_id"] for r in topics if r["level"] == "3"]
    target = np.zeros((len(records), len(leaves)), dtype=int)
    for i, record in enumerate(records):
        row = assignments[record["patent_id"]]
        for field in (["primary_topic_id"] if primary_only else ["primary_topic_id", "secondary_topic_id"]):
            if row[field]:
                target[i, leaves.index(row[field])] = 1
    return target


def topic_fit(x, train, target, groups):
    def classifier():
        return OneVsRestClassifier(LogisticRegression(C=2., solver="liblinear", max_iter=250))
    fitted = classifier().fit(x[train], target[train])
    p = fitted.predict_proba(x)
    fold_rows = []
    for fold, (inside, outside) in enumerate(GroupKFold(3).split(train, target[train], groups[train])):
        model = classifier().fit(x[train[inside]], target[train[inside]])
        p[train[outside]] = model.predict_proba(x[train[outside]])
        assert not set(groups[train[inside]]) & set(groups[train[outside]])
        fold_rows.append({"fold": fold, "fit_records": len(inside), "heldout_records": len(outside),
                          "fit_positive_counts": target[train[inside]].sum(axis=0).tolist()})
    return fitted, p, fold_rows


def ancestor_matrices(topics):
    lookup = {r["topic_id"]: r for r in topics}
    leaves = [r["topic_id"] for r in topics if r["level"] == "3"]
    matrices = []
    for level in [1, 2, 3]:
        nodes = [r["topic_id"] for r in topics if int(r["level"]) == level]
        matrix = np.zeros((len(leaves), len(nodes)))
        for i, leaf in enumerate(leaves):
            node = leaf
            while int(lookup[node]["level"]) > level:
                node = lookup[node]["parent_topic_id"]
            matrix[i, nodes.index(node)] = 1.
        matrices.append(matrix)
    return matrices


def unions(p, matrices):
    return [-np.expm1(np.log1p(-np.clip(p, 1e-7, 1-1e-7)) @ a) for a in matrices]


def overlap(a, b):
    return np.mean([1-np.prod(1-x*y, axis=1) for x, y in zip(a, b)], axis=0)


def entropy(p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return -(p*np.log(p)+(1-p)*np.log1p(-p)).mean(axis=1) / np.log(2)


def rowsim(a, b):
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def lexical(views, q, c):
    return np.mean([rowsim(v[q], v[c]) for v in views], axis=0)


def label_function(assignments, topics):
    tree = {r["topic_id"]: r for r in topics}
    paths = {}
    for pid, assignment in assignments.items():
        ancestors = set()
        for field in ["primary_topic_id", "secondary_topic_id"]:
            node = assignment[field]
            while node:
                ancestors.add(node)
                node = tree[node]["parent_topic_id"]
        paths[pid] = ancestors
    def grade(a, b):
        return max(int(tree[t]["level"]) for t in paths[a] & paths[b])
    return grade


def prepare_pool(records, partition, rep, seed, grade):
    rng = np.random.default_rng(seed + 1000)
    components = np.array([r["component_id"] for r in records])
    result = []
    for name in PARTITIONS:
        candidates = np.flatnonzero(partition == name)
        queries = rng.permutation(candidates)[:QUERY_LIMITS[name]]
        for start in range(0, len(queries), 48):
            batch = queries[start:start+48]
            similarities = (rep.x[batch] @ rep.x[candidates].T).toarray()
            for q, sim in zip(batch, similarities):
                eligible = components[candidates] != components[q]
                sim[~eligible] = -np.inf
                nearest_local = np.lexsort((candidates, -sim))[:20]
                nearest = candidates[nearest_local]
                assert eligible[nearest_local].all()
                remaining = candidates[eligible & ~np.isin(candidates, nearest)]
                random_candidates = rng.choice(remaining, 20, replace=False)
                for c, source in [(int(c), "lexical20") for c in nearest] + [(int(c), "random20") for c in random_candidates]:
                    result.append({"split_seed": seed, "partition": name,
                                   "query_id": records[q]["patent_id"], "candidate_id": records[c]["patent_id"],
                                   "query_component": components[q], "candidate_component": components[c],
                                   "query_family": records[q]["family_id"],
                                   "candidate_source": source, "grade": grade(records[q]["patent_id"], records[c]["patent_id"])})
    assert all(n == 40 for n in Counter(r["query_id"] for r in result).values())
    assert all(r["query_component"] != r["candidate_component"] for r in result)
    assert len({(r["query_id"], r["candidate_id"]) for r in result}) == len(result)
    return result


def mapped_neighbors(records, partition, mapped_edges):
    outgoing = defaultdict(list)
    cross = 0
    for a, b in sorted(mapped_edges):
        if partition[a] != partition[b]:
            cross += 1
        else:
            outgoing[a].append(b)
    neighbor = np.repeat(np.arange(len(records))[:, None], 3, axis=1)
    allowed = np.zeros((len(records), 3), dtype=bool)
    extra = 0
    for a, destinations in outgoing.items():
        chosen = destinations[:3]
        neighbor[a, :len(chosen)] = chosen
        allowed[a, :len(chosen)] = True
        extra += max(0, len(destinations)-3)
    for a, b in zip(*np.where(allowed)):
        assert partition[a] == partition[neighbor[a, b]] and a != neighbor[a, b]
    summary = {"cross_partition_edges_masked": cross, "additional_unique_edges_above_cap_discarded": extra,
               "retained_edges": int(allowed.sum()),
               "degree_counts": dict(Counter(allowed.sum(axis=1).tolist())),
               "records_with_any_neighbor": int(allowed.any(axis=1).sum()),
               "coverage_by_partition": {s: {"records": int((partition == s).sum()),
                   "with_neighbor": int(allowed[partition == s].any(axis=1).sum()),
                   "edges": int(allowed[partition == s].sum())} for s in PARTITIONS}}
    return neighbor, allowed, summary


def per_query(scores, pairs):
    groups = defaultdict(list)
    for i, row in enumerate(pairs):
        groups[row["query_id"]].append(i)
    result = []
    scores = np.asarray(scores)
    for query, indices in groups.items():
        idx = np.asarray(indices)
        labels = np.array([int(pairs[i]["grade"]) for i in idx])
        ids = np.array([pairs[i]["candidate_id"] for i in idx])
        order = np.lexsort((ids, -scores[idx]))
        ideal = np.sort(labels)[::-1]
        gain = 2.**labels - 1
        row = {"query_id": query, "component_id": pairs[idx[0]]["query_component"],
               "family_id": pairs[idx[0]]["query_family"], "candidates": len(idx),
               "positive_candidates": int((labels >= 2).sum())}
        for k in [1, 5, 10, 20, 40]:
            n = min(k, len(idx)); discounts = np.log2(np.arange(2, n+2))
            denominator = np.sum((2.**ideal[:n]-1)/discounts)
            row["ndcg"+str(k)] = float(np.sum(gain[order[:n]]/discounts)/denominator) if denominator else np.nan
        relevant = labels[order] >= 2
        found = np.flatnonzero(relevant)
        row["mrr"] = float(1/(found[0]+1)) if len(found) else 0.
        row["precision3"] = float(relevant[:3].mean())
        row["precision10"] = float(relevant[:10].mean())
        row["recall10_pool"] = float(relevant[:10].sum() / relevant.sum()) if relevant.sum() else np.nan
        result.append(row)
    return result


def aggregate_queries(rows):
    keys = ["ndcg1", "ndcg5", "ndcg10", "ndcg20", "ndcg40", "mrr", "precision3", "precision10", "recall10_pool"]
    return {"queries": len(rows), "pairs": sum(r["candidates"] for r in rows),
            "queries_with_positive_gain": sum(np.isfinite(r["ndcg10"]) for r in rows),
            **{k: float(np.nanmean([r[k] for r in rows])) for k in keys}}


def validation_score(scores, pairs):
    metrics = aggregate_queries(per_query(scores, pairs))
    y = np.array([int(r["grade"])/3 for r in pairs])
    return metrics["ndcg10"], -float(np.mean(abs(np.clip(np.asarray(scores), 0, 1)-y)))


def bootstrap_interval(rows, metric="ndcg10", cluster="component_id", count=1500, seed=20261030):
    grouped = defaultdict(list)
    for row in rows:
        if np.isfinite(row[metric]):
            grouped[row[cluster]].append(row[metric])
    sums = np.array([sum(v) for v in grouped.values()]); sizes = np.array([len(v) for v in grouped.values()])
    if not len(sums):
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(sums), (count, len(sums)))
    values = sums[choices].sum(axis=1)/sizes[choices].sum(axis=1)
    return np.quantile(values, [.025, .975]).tolist()


def random_baseline(pairs, seed):
    groups = defaultdict(list)
    for row in pairs:
        groups[row["query_id"]].append(row)
    rng = np.random.default_rng(seed + 7000)
    output = []
    for query, rows in groups.items():
        labels = np.array([int(r["grade"]) for r in rows])
        permutations = np.argsort(rng.random((1000, len(rows))), axis=1)
        discounts = np.log2(np.arange(2, 12))
        ideal = np.sum((2.**np.sort(labels)[::-1][:10]-1)/discounts)
        values = ((2.**labels[permutations[:, :10]]-1)/discounts).sum(axis=1)/ideal if ideal else np.full(1000, np.nan)
        exact = float((2.**labels-1).mean() * np.sum(1/discounts)/ideal) if ideal else np.nan
        output.append({"query_id": query, "component_id": rows[0]["query_component"],
                       "family_id": rows[0]["query_family"], "partition": rows[0]["partition"],
                       "candidates": len(rows), "random_permutations": 1000,
                       "ndcg10": exact, "monte_carlo_ndcg10": float(np.nanmean(values)),
                       "permutation_sd": float(np.nanstd(values)),
                       "permutation_low": float(np.nanquantile(values, .025)),
                       "permutation_high": float(np.nanquantile(values, .975))})
    return output


def pair_features(z, q, c):
    a, b = z[q], z[c]
    return np.column_stack([a, b, abs(a-b), a*b])


def flat_pair_features(p, q, c, lex):
    return np.column_stack([p[q]*p[c], abs(p[q]-p[c]), lex])


def graph_features(views, p, q, c, neighbor, matrices):
    u, ent = unions(p, matrices), entropy(p)
    lex = lexical(views, q, c)
    hierarchy = overlap([a[q] for a in u], [a[c] for a in u])
    ln = np.column_stack([lexical(views, q, neighbor[:, j]) for j in range(3)])
    hn = np.column_stack([overlap([a[q] for a in u], [a[neighbor[:, j]] for a in u]) for j in range(3)])
    cn = np.column_stack([lexical(views, c, neighbor[:, j]) for j in range(3)])
    x = np.stack([np.ones_like(ln), ln, cn, abs(hn-hierarchy[:, None]),
                  np.repeat(ent[c, None], 3, axis=1), ent[neighbor]], axis=2)
    return lex, hierarchy, ln, hn, x


def bm25_scores(records, train, q, c):
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 2), min_df=2,
                                 max_features=12000, dtype=np.float64)
    texts = [r["model_text"] for r in records]
    vectorizer.fit([texts[i] for i in train]); counts = vectorizer.transform(texts).tocsr()
    df = np.asarray((counts[train] > 0).sum(axis=0)).ravel()
    idf = np.log(1+(len(train)-df+.5)/(df+.5))
    lengths = np.asarray(counts.sum(axis=1)).ravel(); avg = lengths[train].mean()
    norms = 1.2*(.25+.75*lengths/avg)
    weighted = counts.copy()
    for i in range(len(records)):
        sl = slice(weighted.indptr[i], weighted.indptr[i+1]); f = weighted.data[sl]
        weighted.data[sl] = f*2.2/(f+norms[i])*idf[weighted.indices[sl]]
    binary = counts.copy(); binary.data[:] = 1.
    return rowsim(binary[q], weighted[c])


def fit_models(records, assignments, topics, partition, pairs, neighbor, permitted, rep, seed):
    begin = time.perf_counter()
    ids = {r["patent_id"]: i for i, r in enumerate(records)}
    train = np.flatnonzero(partition == "train")
    components = np.array([r["component_id"] for r in records])
    q = np.array([ids[r["query_id"]] for r in pairs]); c = np.array([ids[r["candidate_id"]] for r in pairs])
    pair_partition = np.array([r["partition"] for r in pairs])
    tr, va, te = [pair_partition == name for name in PARTITIONS]
    valpairs = [r for r in pairs if r["partition"] == "validation"]
    y = np.array([int(r["grade"])/3 for r in pairs])
    target = topic_targets(records, assignments, topics)
    topic, p, folds = topic_fit(rep.x, train, target, components)
    primary_target = topic_targets(records, assignments, topics, True)
    primary, primary_p, primary_folds = topic_fit(rep.x, train, primary_target, components)
    print(seed, "topics fitted", flush=True)
    matrices = ancestor_matrices(topics)
    lex, hierarchy, ln, hn, raw_x = graph_features(rep.views, p, q, c, neighbor[c], matrices)
    flat = 1-np.prod(1-p[q]*p[c], axis=1)
    primary_u = unions(primary_p, matrices)
    primary_h = overlap([a[q] for a in primary_u], [a[c] for a in primary_u])
    pred = {"TF-IDF": lex, "BM25": bm25_scores(records, train, q, c), "Flat leaf": flat}
    selection, config_rows = {}, []
    def select_blend(name, signal):
        grid = []
        for alpha in ALPHAS:
            score = (1-alpha)*lex+alpha*signal
            ndcg, neg_mae = validation_score(score[va], valpairs)
            grid.append({"method": name, "alpha": alpha, "rho": 0., "ndcg10": ndcg, "mae": -neg_mae})
        best = max(grid, key=lambda r: (r["ndcg10"], -r["mae"]))
        config_rows.extend(grid); selection[name] = best
        pred[name] = (1-best["alpha"])*lex+best["alpha"]*signal
    select_blend("Flat blend", flat)
    select_blend("Hierarchy", hierarchy)
    select_blend("Primary hierarchy", primary_h)
    uniform_grid = []
    allowed_pairs = permitted[c]
    for alpha in ALPHAS:
        base, bn = (1-alpha)*lex+alpha*hierarchy, (1-alpha)*ln+alpha*hn
        for rho in RHOS:
            score = (1-rho)*base+rho*(base+(bn*allowed_pairs).sum(axis=1))/(1+allowed_pairs.sum(axis=1))
            ndcg, neg_mae = validation_score(score[va], valpairs)
            uniform_grid.append({"method": "Uniform", "alpha": alpha, "rho": rho, "ndcg10": ndcg, "mae": -neg_mae})
    best = max(uniform_grid, key=lambda r: (r["ndcg10"], -r["mae"]))
    selection["Uniform"] = best; config_rows.extend(uniform_grid)
    base = (1-best["alpha"])*lex+best["alpha"]*hierarchy
    bn = (1-best["alpha"])*ln+best["alpha"]*hn
    pred["Uniform"] = (1-best["rho"])*base+best["rho"]*(base+(bn*allowed_pairs).sum(axis=1))/(1+allowed_pairs.sum(axis=1))
    flat_x = flat_pair_features(p, q, c, lex)
    flat_scaler = StandardScaler().fit(flat_x[tr]); flat_x = flat_scaler.transform(flat_x)
    flat_grid = []; flat_models = []
    for penalty in [.1, 1., 10., 100.]:
        fitted = Ridge(alpha=penalty).fit(flat_x[tr], y[tr])
        scores = fitted.predict(flat_x)
        ndcg, neg_mae = validation_score(scores[va], valpairs)
        flat_models.append((fitted, scores)); flat_grid.append({"method": "Flat pair ridge", "ridge_alpha": penalty,
            "ndcg10": ndcg, "mae": -neg_mae})
    chosen = max(range(len(flat_grid)), key=lambda i: (flat_grid[i]["ndcg10"], -flat_grid[i]["mae"]))
    flat_model, pred["Flat pair ridge"] = flat_models[chosen]
    selection["Flat pair ridge"] = flat_grid[chosen]; config_rows.extend(flat_grid)
    del flat_models, flat_x
    svd = TruncatedSVD(n_components=64, random_state=17).fit(rep.x[train])
    z = svd.transform(rep.x)
    xp = pair_features(z, q, c)
    scaler = StandardScaler().fit(xp[tr]); xp = scaler.transform(xp)
    neural_grid = []; neural_models = []
    for widths in [(64, 32), (128, 64)]:
        for penalty in [.001, .01, .1]:
            config = {"hidden_layer_sizes": widths, "alpha": penalty}
            t0 = time.perf_counter()
            fitted = MLPRegressor(**config, activation="relu", learning_rate_init=.001, batch_size=256,
                max_iter=80, early_stopping=True, validation_fraction=.1, n_iter_no_change=7, random_state=17)
            with warnings.catch_warnings(record=True) as caught:
                fitted.fit(xp[tr], y[tr])
            scores = fitted.predict(xp)
            ndcg, neg_mae = validation_score(scores[va], valpairs)
            neural_grid.append({"method": "MLP", **config, "ndcg10": ndcg, "mae": -neg_mae,
                                "fit_seconds": time.perf_counter()-t0, "iterations": fitted.n_iter_,
                                "warnings": [str(w.message) for w in caught]})
            neural_models.append((fitted, scores))
            print(seed, "MLP config", widths, penalty, fitted.n_iter_, flush=True)
    chosen = max(range(len(neural_grid)), key=lambda i: (neural_grid[i]["ndcg10"], -neural_grid[i]["mae"]))
    selection["MLP"] = neural_grid[chosen]; config_rows.extend(neural_grid)
    mlps = [neural_models[chosen][0]]; mlp_scores = [neural_models[chosen][1]]
    selected_config = {k: neural_grid[chosen][k] for k in ["hidden_layer_sizes", "alpha"]}
    for model_seed in [29, 43]:
        fitted = MLPRegressor(**selected_config, activation="relu", learning_rate_init=.001, batch_size=256,
            max_iter=80, early_stopping=True, validation_fraction=.1, n_iter_no_change=7, random_state=model_seed)
        fitted.fit(xp[tr], y[tr]); mlps.append(fitted)
        mlp_scores.append(fitted.predict(xp))
    pred["MLP"] = np.mean(mlp_scores, axis=0)
    del neural_models, mlp_scores
    rank_grid = []; rank_models = []
    for depth in [3, 5]:
        for estimators in [100, 200]:
            model = xgb.XGBRanker(objective="rank:ndcg", tree_method="hist", max_depth=depth,
                n_estimators=estimators, learning_rate=.05, min_child_weight=5,
                subsample=.8, colsample_bytree=.8, max_bin=128, n_jobs=1, random_state=17)
            t0 = time.perf_counter()
            model.fit(xp[tr], (3*y[tr]).astype(int), group=np.repeat(40, tr.sum()//40), verbose=False)
            scores = model.predict(xp)
            ndcg, neg_mae = validation_score(scores[va], valpairs)
            rank_grid.append({"method": "LambdaMART", "max_depth": depth, "n_estimators": estimators,
                              "ndcg10": ndcg, "mae": -neg_mae, "fit_seconds": time.perf_counter()-t0})
            rank_models.append((model, scores))
            print(seed, "LambdaMART", depth, estimators, flush=True)
    chosen = max(range(len(rank_grid)), key=lambda i: (rank_grid[i]["ndcg10"], -rank_grid[i]["mae"]))
    rank_model, pred["LambdaMART"] = rank_models[chosen]
    selection["LambdaMART"] = rank_grid[chosen]; config_rows.extend(rank_grid)
    del rank_models, xp
    selected_edges = raw_x[tr][allowed_pairs[tr]]
    center = selected_edges.mean(axis=0); scale = selected_edges.std(axis=0)
    center[0] = 0.; scale[0] = 1.; scale[scale < 1e-6] = 1.
    x = (raw_x-center)/scale
    gate_models = {}
    for objective in ["expected", "mean", "listnet"]:
        grid, fits = [], {}
        for alpha in ALPHAS:
            base = (1-alpha)*lex+alpha*hierarchy; bn = (1-alpha)*ln+alpha*hn
            fitted = fit_gate(x[tr], base[tr], bn[tr], y[tr], rho=.3, objective=objective,
                              groups=np.array([r["query_id"] for r in pairs])[tr],
                              allowed=allowed_pairs[tr], maxiter=100)
            fits[alpha] = fitted
            for rho in RHOS:
                scores = evaluate_gate(x[va], base[va], bn[va], fitted["theta"], rho, allowed_pairs[va])[0]
                ndcg, neg_mae = validation_score(scores, valpairs)
                grid.append({"method": "Gate "+objective, "alpha": alpha, "rho": rho,
                             "ndcg10": ndcg, "mae": -neg_mae, "optimizer_success": fitted["success"],
                             "optimizer_message": fitted["message"], "train_loss": fitted["fun"]})
        best = max(grid, key=lambda r: (r["ndcg10"], -r["mae"]))
        name = "Gate "+objective; selection[name] = best; config_rows.extend(grid)
        fitted = fits[best["alpha"]]
        base = (1-best["alpha"])*lex+best["alpha"]*hierarchy
        bn = (1-best["alpha"])*ln+best["alpha"]*hn
        pred[name], gate_prob, gate_var = evaluate_gate(x, base, bn, fitted["theta"], best["rho"], allowed_pairs)
        gate_models[name] = {"theta": fitted["theta"], "alpha": best["alpha"], "rho": best["rho"]}
        effective_test = gate_prob[te][allowed_pairs[te]]
        selection[name].update({"theta": fitted["theta"], "test_expected_edges": gate_prob[te].sum(axis=1).mean(),
                                "test_mask_variance": gate_var[te].mean(),
                                "test_allowed_probability_below_01": float(np.mean(effective_test < .01)),
                                "test_allowed_probability_above_99": float(np.mean(effective_test > .99)),
                                "test_allowed_probability_quantiles": np.quantile(effective_test, [0, .25, .5, .75, 1])})
        print(seed, name, best["alpha"], best["rho"], flush=True)
    state = {"seed": seed, "selection": selection, "topic_oof_folds": folds,
             "primary_topic_oof_folds": primary_folds, "training_leaf_positive_counts": target[train].sum(axis=0),
             "training_leaf_coverage": int((target[train].sum(axis=0) > 0).sum()),
             "vocabulary_sizes": [len(v.vocabulary_) for v in rep.vectorizers],
             "fit_seconds": time.perf_counter()-begin,
             "gate_training_rho": .3, "selection_rule": "validation nDCG@10 on raw ranking scores; lower clipped normalized score MAE breaks ties",
             "ranking_scores": "unclipped Ridge and mean of three raw MLP outputs; raw LambdaMART; bounded text/topic/graph scores",
             "hyperparameter_budget": {"MLP": 6, "LambdaMART": 4, "Flat pair ridge": 4,
                 "Hierarchy": 5, "Primary hierarchy": 5, "Flat blend": 5, "Uniform": 20,
                 "Gate expected": "5 fits at rho=.3 then 20 validation configurations", "Gate mean": "same", "Gate listnet": "same"},
             "gate_feature_mean": center, "gate_feature_scale": scale}
    write_json(f"selection_seed{seed}.json", state)
    # Heterogeneous grids remain JSON, preserving each model's own full settings.
    write_json(f"validation_grid_seed{seed}.json", config_rows)
    context = {"rep": rep, "topic": topic, "primary": primary, "p": p, "primary_p": primary_p,
               "matrices": matrices, "u": unions(p, matrices), "entropy": entropy(p),
               "svd": svd, "z": z, "scaler": scaler, "mlps": mlps,
               "flat_scaler": flat_scaler, "flat_model": flat_model,
               "selection": selection, "gate_models": gate_models, "center": center, "scale": scale,
               "q": q, "c": c, "pair_partition": pair_partition,
               "neighbor": neighbor, "permitted": permitted}
    return pred, context, state


def summarize_predictions(predictions, pairs, seed):
    partition = np.array([r["partition"] for r in pairs])
    metrics, query_rows, queries_by_method = [], [], {}
    for name, scores in predictions.items():
        assert len(scores) == len(pairs) and np.isfinite(scores).all(), name
        for part in PARTITIONS:
            mask = partition == part
            rows = per_query(np.asarray(scores)[mask], [r for r, keep in zip(pairs, mask) if keep])
            queries_by_method[(name, part)] = rows
            metrics.append({"split_seed": seed, "partition": part, "method": name,
                **aggregate_queries(rows),
                "component_ci_low": bootstrap_interval(rows)[0], "component_ci_high": bootstrap_interval(rows)[1],
                "family_ci_low": bootstrap_interval(rows, cluster="family_id")[0],
                "family_ci_high": bootstrap_interval(rows, cluster="family_id")[1]})
            query_rows.extend({"split_seed": seed, "partition": part, "method": name, **r} for r in rows)
    write_csv(f"metrics_seed{seed}.csv", metrics)
    write_csv(f"query_scores_seed{seed}.csv", query_rows)
    write_csv(f"pair_scores_seed{seed}.csv", ({**row, **{name: float(scores[i]) for name, scores in predictions.items()}}
        for i, row in enumerate(pairs)))
    differences = []
    for name in ["Gate expected", "Gate mean", "Gate listnet"]:
        for comparator in ["Hierarchy", "Uniform", "Flat pair ridge", "Flat blend", "MLP", "LambdaMART"]:
            for part in PARTITIONS:
                left, right = queries_by_method[(name, part)], queries_by_method[(comparator, part)]
                assert [r["query_id"] for r in left] == [r["query_id"] for r in right]
                delta = [{**a, "ndcg10": a["ndcg10"]-b["ndcg10"]} for a, b in zip(left, right)]
                for cluster in ["component_id", "family_id"]:
                    low, high = bootstrap_interval(delta, cluster=cluster)
                    differences.append({"split_seed": seed, "partition": part, "method": name,
                        "comparator": comparator, "resampling_unit": cluster, "queries": len(delta),
                        "difference": float(np.nanmean([r["ndcg10"] for r in delta])), "low": low, "high": high})
    write_csv(f"paired_differences_seed{seed}.csv", differences)
    random_rows = random_baseline(pairs, seed)
    write_csv(f"random_query_baseline_seed{seed}.csv", ({"split_seed": seed, **r} for r in random_rows))
    write_csv(f"random_metrics_seed{seed}.csv", ({"split_seed": seed, "partition": part,
        "queries": sum(r["partition"] == part for r in random_rows), "permutations_per_query": 1000,
        "ndcg10": float(np.nanmean([r["ndcg10"] for r in random_rows if r["partition"] == part]))} for part in PARTITIONS))
    return metrics


def stress_test(records, pairs, context, predictions, seed):
    indices = np.flatnonzero(context["pair_partition"] == "test")
    testpairs = [pairs[i] for i in indices]
    q, c = context["q"][indices], context["c"][indices]
    unique_queries = np.unique(q); query_map = {int(k): i for i, k in enumerate(unique_queries)}
    local_q = np.array([query_map[int(i)] for i in q])
    neighbors = context["neighbor"][c]; allowed = context["permitted"][c]
    selection = context["selection"]; rep = context["rep"]
    original_views = rep.views; original_u = context["u"]
    rows = []
    stress_methods = ["TF-IDF", "MLP", "Flat pair ridge", "Hierarchy", "Gate expected", "Gate mean", "Gate listnet"]
    for noise_seed in SEEDS:
        rng = np.random.default_rng(noise_seed)
        noisy = []
        for i in unique_queries:
            record = dict(records[i])
            for field in ["title", "abstract", "claims"]:
                text = record[field]
                record[field] = "".join(ch for ch, retain in zip(text, rng.random(len(text)) >= .5) if retain)
            noisy.append(record)
        query_views, query_x = rep.transform(noisy)
        qp = context["topic"].predict_proba(query_x)
        qu = unions(qp, context["matrices"])
        def qlex(documents):
            return np.mean([rowsim(a[local_q], b[documents]) for a, b in zip(query_views, original_views)], axis=0)
        lex = qlex(c)
        hierarchy = overlap([a[local_q] for a in qu], [a[c] for a in original_u])
        score = {"TF-IDF": lex}
        aa = selection["Hierarchy"]["alpha"]
        score["Hierarchy"] = (1-aa)*lex+aa*hierarchy
        flat_x = np.column_stack([qp[local_q]*context["p"][c], abs(qp[local_q]-context["p"][c]), lex])
        score["Flat pair ridge"] = context["flat_model"].predict(context["flat_scaler"].transform(flat_x))
        qz = context["svd"].transform(query_x)[local_q]; cz = context["z"][c]
        xp = context["scaler"].transform(np.column_stack([qz, cz, abs(qz-cz), qz*cz]))
        score["MLP"] = np.mean([m.predict(xp) for m in context["mlps"]], axis=0)
        ln = np.column_stack([qlex(neighbors[:, j]) for j in range(3)])
        hn = np.column_stack([overlap([a[local_q] for a in qu], [a[neighbors[:, j]] for a in original_u]) for j in range(3)])
        cn = np.column_stack([lexical(original_views, c, neighbors[:, j]) for j in range(3)])
        ent = context["entropy"]
        features = np.stack([np.ones_like(ln), ln, cn, abs(hn-hierarchy[:, None]),
            np.repeat(ent[c, None], 3, axis=1), ent[neighbors]], axis=2)
        features = (features-context["center"])/context["scale"]
        for name, model in context["gate_models"].items():
            aa, rr = model["alpha"], model["rho"]
            score[name] = evaluate_gate(features, (1-aa)*lex+aa*hierarchy,
                (1-aa)*ln+aa*hn, model["theta"], rr, allowed)[0]
        for name in stress_methods:
            pq = per_query(score[name], testpairs)
            summary = aggregate_queries(pq)
            rows.append({"split_seed": seed, "noise_seed": noise_seed, "query_deletion_rate": .5,
                         "method": name, **summary})
    write_csv(f"stress50_seed{seed}.csv", rows)


def aggregate_all():
    all_rows = []
    for seed in SEEDS:
        path = OUT / f"metrics_seed{seed}.csv"
        if path.exists():
            all_rows.extend(csv.DictReader(path.open()))
    if not all_rows:
        return
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["partition"], row["method"])].append(row)
    output = []
    for (part, method), rows in grouped.items():
        item = {"partition": part, "method": method, "split_runs": len(rows)}
        for metric in ["ndcg10", "mrr", "precision3", "precision10", "recall10_pool"]:
            values = np.array([float(r[metric]) for r in rows])
            item[metric+"_mean"] = values.mean(); item[metric+"_sd"] = values.std(ddof=1) if len(values) > 1 else 0.
            item[metric+"_min"] = values.min(); item[metric+"_max"] = values.max()
        output.append(item)
    write_csv("aggregate_metrics.csv", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    records, assignments, topics, mapped_edges, audit = audit_data()
    grade = label_function(assignments, topics)
    manifest = []
    for seed in args.seeds:
        partition = split_records(records, seed)
        for record, part in zip(records, partition):
            manifest.append({"split_seed": seed, "patent_id": record["patent_id"],
                             "component_id": record["component_id"], "partition": part,
                             "source_family_id": record["family_id"]})
    existing = OUT / "partition_manifest.csv"
    if existing.exists():
        manifest = [r for r in csv.DictReader(existing.open()) if int(r["split_seed"]) not in args.seeds] + manifest
    write_csv("partition_manifest.csv", manifest)
    for seed in args.seeds:
        t0 = time.perf_counter()
        partition = split_records(records, seed)
        train = np.flatnonzero(partition == "train")
        rep = Representation().fit_text(records, train)
        pairs = prepare_pool(records, partition, rep, seed, grade)
        write_csv(f"pools_seed{seed}.csv", pairs)
        neighbor, allowed, graph_summary = mapped_neighbors(records, partition, mapped_edges)
        write_csv(f"neighbors_seed{seed}.csv", ({"patent_id": row["patent_id"], "partition": partition[i],
            "neighbor_1": records[neighbor[i, 0]]["patent_id"] if allowed[i, 0] else "",
            "neighbor_2": records[neighbor[i, 1]]["patent_id"] if allowed[i, 1] else "",
            "neighbor_3": records[neighbor[i, 2]]["patent_id"] if allowed[i, 2] else ""} for i, row in enumerate(records)))
        target = topic_targets(records, assignments, topics)
        split_summary = {"seed": seed, "partitions": {}, "graph": graph_summary,
                         "leaves": [r["topic_id"] for r in topics if r["level"] == "3"]}
        for name in PARTITIONS:
            selected = np.flatnonzero(partition == name)
            split_summary["partitions"][name] = {"records": len(selected),
                "components": len({records[i]["component_id"] for i in selected}),
                "source_families": len({records[i]["family_id"] for i in selected}),
                "queries": len({r["query_id"] for r in pairs if r["partition"] == name}),
                "pairs": sum(r["partition"] == name for r in pairs),
                "leaf_positive_counts": target[selected].sum(axis=0),
                "leaf_coverage": int((target[selected].sum(axis=0) > 0).sum()),
                "grade_counts": dict(Counter(r["grade"] for r in pairs if r["partition"] == name))}
        assert len({r["component_id"] for r in records}) == sum(v["components"] for v in split_summary["partitions"].values())
        write_json(f"split_audit_seed{seed}.json", split_summary)
        print(seed, "PLAN_READY", {k: {a: b for a, b in v.items() if a in ["records", "components", "queries", "pairs", "leaf_coverage"]}
              for k, v in split_summary["partitions"].items()}, flush=True)
        if args.prepare_only:
            continue
        predictions, context, state = fit_models(records, assignments, topics, partition, pairs, neighbor, allowed, rep, seed)
        metrics = summarize_predictions(predictions, pairs, seed)
        stress_test(records, pairs, context, predictions, seed)
        aggregate_all()
        print(seed, "COMPLETE", "seconds", round(time.perf_counter()-t0, 2), flush=True)
        print(json.dumps([r for r in metrics if r["partition"] == "test"], default=json_default), flush=True)
    for name, expected in audit["input_sha256"].items():
        assert hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest() == expected, "Input changed during run"
    write_json("pipeline_status.json", {"requested_seeds": args.seeds, "prepare_only": args.prepare_only,
        "seconds": time.perf_counter()-begin, "input_sha256": audit["input_sha256"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__, "xgboost": xgb.__version__},
        "threads": 1, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
