#!/usr/bin/env python3
"""Locally encode official BGE-M3 dense vectors and score supplied reranking pools.

Download the pinned public checkpoint separately. This runner is offline and never
uploads corpus text. It caches text-keyed embeddings and can resume across pools.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
HF_CACHE = ROOT / ".cache_review/huggingface"
for key, value in {
    "HF_HOME": HF_CACHE,
    "HF_HUB_CACHE": HF_CACHE / "hub",
    "TRANSFORMERS_CACHE": HF_CACHE / "transformers",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ[key] = str(value)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document_text(record, abstract_prestripped=False):
    """Use the three observed text fields and remove the direct abstract prefix."""
    for column in ("stripped_abstract", "abstract_clean", "residual_abstract", "abstract_stripped"):
        if column in record:
            abstract = record[column]
            break
    else:
        if abstract_prestripped:
            return "\n".join([record["title"], record["abstract"], record["claims"]])
        abstract = record["abstract"].split("。", 1)[1] if "。" in record["abstract"] else ""
    return "\n".join([record["title"], abstract, record["claims"]])


def evaluate(rows):
    """Macro-query graded ranking; exact ties are broken by candidate identifier."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    output = []
    for query, pool in groups.items():
        ranked = sorted(pool, key=lambda r: (-float(r["bge_score"]), r["candidate_id"]))
        ideal = sorted((int(r["grade"]) for r in pool), reverse=True)
        result = {"query_id": query, "candidates": len(pool)}
        for cutoff in (1, 3, 5, 10):
            limit = min(cutoff, len(pool))
            idcg = math.fsum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:limit]))
            dcg = math.fsum((2 ** int(r["grade"]) - 1) / math.log2(i + 2)
                            for i, r in enumerate(ranked[:limit]))
            result[f"ndcg{cutoff}"] = dcg / idcg if idcg else float("nan")
        relevant_ranks = [i for i, r in enumerate(ranked) if int(r["grade"]) >= 2]
        result["mrr"] = 1 / (relevant_ranks[0] + 1) if relevant_ranks else 0.
        top = ranked[:min(3, len(ranked))]
        result["precision3"] = sum(int(r["grade"]) >= 2 for r in top) / len(top)
        output.append(result)
    summary = {
        "method": "BGE-M3 zero-shot dense",
        "queries": len(output),
        "pairs": len(rows),
        "min_candidates": min(len(p) for p in groups.values()),
        "max_candidates": max(len(p) for p in groups.values()),
    }
    for metric in ("ndcg1", "ndcg3", "ndcg5", "ndcg10", "mrr", "precision3"):
        values = [r[metric] for r in output if math.isfinite(r[metric])]
        summary[metric] = math.fsum(values) / len(values) if values else float("nan")
    summary["queries_with_positive_gain"] = sum(math.isfinite(r["ndcg10"]) for r in output)
    return summary, output


def main(args):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "bge_run_summary.json"
    environment = {
        "python": sys.version,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_path": torch.__file__,
        "transformers": transformers.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }
    device = args.device
    if device == "auto":
        device = "mps" if environment["mps_available"] else "cpu"
    if device == "mps" and not environment["mps_available"]:
        raise RuntimeError("MPS requested but unavailable in this process; no silent CPU fallback")
    torch.set_num_threads(args.threads)
    torch.manual_seed(17)
    model_path = (HF_CACHE / "hub/models--BAAI--bge-m3/snapshots" / REVISION).resolve()
    if not (model_path / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"Pinned BGE-M3 checkpoint not downloaded: {model_path}")
    pooling = json.loads((model_path / "1_Pooling/config.json").read_text())
    if not pooling.get("pooling_mode_cls_token") or pooling.get("word_embedding_dimension") != 1024:
        raise ValueError("Official checkpoint pooling metadata does not match CLS/1024 dense encoding")
    records = read_csv(args.representatives)
    id_column = next((c for c in ("patent_id", "representative_id", "record_id") if c in records[0]), None)
    if id_column is None:
        raise ValueError(f"No known representative ID column: {list(records[0])}")
    id_to_key = {}
    texts = {}
    for record in records:
        text = document_text(record, args.abstract_prestripped)
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        id_to_key[record[id_column]] = key
        texts[key] = text
    pool_files = [Path(p).resolve() for p in args.pools]
    pool_sets = []
    needed_ids = set()
    for path in pool_files:
        pool = [r for r in read_csv(path) if r["partition"] in args.partitions]
        if not pool:
            raise ValueError(f"No requested partitions in {path}")
        pool_sets.append((path, pool))
        needed_ids.update(r[k] for r in pool for k in ("query_id", "candidate_id"))
    missing = needed_ids - set(id_to_key)
    if missing:
        raise ValueError(f"{len(missing)} candidate/query IDs absent from representatives")
    needed_keys = set(texts) if args.encode_all else {id_to_key[i] for i in needed_ids}
    cache_path = output / f"bge_dense_embeddings_max{args.max_length}.npz"
    embeddings = {}
    lengths = {}
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["revision"].item()) != REVISION:
                raise ValueError("Embedding cache model revision mismatch")
            embeddings.update(zip(cached["keys"].tolist(), cached["vectors"]))
            lengths.update(zip(cached["keys"].tolist(), cached["lengths"].tolist()))
    pending = sorted(needed_keys - set(embeddings))
    run = {
        "status": "encoding",
        "model": MODEL_ID,
        "revision": REVISION,
        "model_snapshot": str(model_path),
        "checkpoint_sha256": digest(model_path / "pytorch_model.bin"),
        "official_pooling_config": pooling,
        "device": device,
        "environment": environment,
        "precision": "float32",
        "dense_representation": "XLM-R encoder last_hidden_state[:,0], L2 normalized; official CLS pooling config",
        "training": "zero-shot; no local fitting, labels or hyperparameter selection",
        "text": "title + newline + abstract after first Chinese full stop + newline + claims; prestripped abstract column used when supplied",
        "abstract_prestripped": args.abstract_prestripped,
        "max_length": args.max_length,
        "truncation": "right truncation at tokenizer max_length including special tokens; dynamic batch padding",
        "batch_size": args.batch_size,
        "threads": args.threads,
        "network": "HF_HUB_OFFLINE=1; TRANSFORMERS_OFFLINE=1; local checkpoint only; no corpus upload",
        "partitions": args.partitions,
        "representative_records": len(records),
        "unique_three_field_texts": len(texts),
        "needed_endpoint_ids": len(needed_ids),
        "requested_unique_texts": len(needed_keys),
        "cached_unique_texts_at_start": len(embeddings),
        "pending_unique_texts_at_start": len(pending),
        "input_sha256": {str(args.representatives): digest(args.representatives),
                          **{str(p): digest(p) for p in pool_files}},
    }
    save_json(summary_path, run)
    print(json.dumps({k: run[k] for k in ("model", "revision", "device", "requested_unique_texts", "pending_unique_texts_at_start")}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    token_start = time.perf_counter()
    # Token counts are measured before truncation to document its actual impact.
    for start in range(0, len(pending), 128):
        keys = pending[start:start + 128]
        encoded = tokenizer([texts[k] for k in keys], truncation=False, padding=False)
        lengths.update({k: len(ids) for k, ids in zip(keys, encoded["input_ids"])})
    run["tokenization_seconds"] = time.perf_counter() - token_start
    token_lengths = [lengths[k] for k in needed_keys]
    run["token_lengths_before_truncation"] = {
        "min": min(token_lengths), "median": float(np.median(token_lengths)),
        "p95": float(np.percentile(token_lengths, 95)), "max": max(token_lengths),
        "truncated_texts": sum(n > args.max_length for n in token_lengths),
    }
    save_json(summary_path, run)
    print(json.dumps(run["token_lengths_before_truncation"]), flush=True)
    load_start = time.perf_counter()
    model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32)
    model.eval().to(device)
    run["model_load_seconds"] = time.perf_counter() - load_start
    print(f"Model loaded on {device} in {run['model_load_seconds']:.2f}s", flush=True)

    def checkpoint():
        keys = sorted(embeddings)
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, revision=np.array(REVISION), keys=np.array(keys),
                            vectors=np.stack([embeddings[k] for k in keys]),
                            lengths=np.array([lengths[k] for k in keys]))
        temporary.replace(cache_path)
        run["cached_unique_texts"] = len(embeddings)
        run["encoded_this_run"] = len(embeddings) - run["cached_unique_texts_at_start"]
        run["encoding_seconds"] = time.perf_counter() - encode_start
        save_json(summary_path, run)

    encode_start = time.perf_counter()
    # Similar lengths share a batch to reduce padding on the GPU.
    pending.sort(key=lambda k: (lengths[k], k))
    with torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            keys = pending[start:start + args.batch_size]
            tokenized = tokenizer([texts[k] for k in keys], padding=True, truncation=True,
                                  max_length=args.max_length, return_tensors="pt")
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            vectors = model(**tokenized, return_dict=True).last_hidden_state[:, 0]
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1).float().cpu().numpy()
            if not np.isfinite(vectors).all():
                raise ValueError("Nonfinite BGE-M3 embeddings")
            embeddings.update(zip(keys, vectors))
            done = min(start + len(keys), len(pending))
            if done == len(pending) or done % args.checkpoint_every < args.batch_size:
                if device == "mps":
                    torch.mps.synchronize()
                checkpoint()
                print(f"Encoded {done}/{len(pending)} new texts; {run['encoding_seconds']:.1f}s", flush=True)
    if not pending:
        run["encoding_seconds"] = 0.
        run["encoded_this_run"] = 0
        run["cached_unique_texts"] = len(embeddings)
    score_start = time.perf_counter()
    vector_norms = [float(np.linalg.norm(embeddings[k])) for k in needed_keys]
    run["embedding_norm_range"] = [min(vector_norms), max(vector_norms)]
    if not all(abs(n - 1.) < 1e-4 for n in vector_norms):
        raise ValueError("Cached BGE vectors are not unit-normalized")
    result_rows = []
    query_rows = []
    for path, pool in pool_sets:
        scored = []
        for row in pool:
            score = float(embeddings[id_to_key[row["query_id"]]] @ embeddings[id_to_key[row["candidate_id"]]])
            scored.append({**row, "bge_score": score})
        write_csv(output / f"bge_scores_{path.stem}.csv", scored)
        groups = defaultdict(list)
        for row in scored:
            groups[(row.get("split_seed", path.stem), row["partition"])].append(row)
        for (seed, partition), subset in groups.items():
            result, per_query = evaluate(subset)
            result_rows.append({"split_seed": seed, "partition": partition, **result})
            query_rows.extend({"split_seed": seed, "partition": partition, **r} for r in per_query)
    write_csv(output / "bge_metrics.csv", result_rows)
    write_csv(output / "bge_query_metrics.csv", query_rows)
    run["status"] = "complete"
    run["scoring_seconds"] = time.perf_counter() - score_start
    run["metrics"] = result_rows
    run["embedding_cache"] = str(cache_path)
    run["input_hashes_unchanged"] = all(digest(p) == h for p, h in run["input_sha256"].items())
    if not run["input_hashes_unchanged"]:
        raise RuntimeError("Input changed during BGE run")
    save_json(summary_path, run)
    print(json.dumps({"status": "complete", "metrics": result_rows}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representatives", type=Path, default=ROOT / "results/revision/representatives.csv")
    parser.add_argument("--pools", nargs="+", required=True)
    parser.add_argument("--partitions", nargs="+", default=["validation", "test"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/revision")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--encode-all", action="store_true")
    parser.add_argument("--abstract-prestripped", action="store_true",
                        help="The input abstract column already excludes its first sentence; do not strip again")
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(arguments.output_dir / "bge_failure.json", {
            "model": MODEL_ID, "revision": REVISION, "error": repr(error),
            "traceback": traceback.format_exc(), "status": "failed_no_result_claim",
        })
        raise
