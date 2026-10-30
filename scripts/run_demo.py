#!/usr/bin/env python3
"""Run the complete local ranking pipeline on newly generated synthetic texts.

DEMO_ONLY: generated records and scores are unrelated to the paper's corpus or
reported measurements. No original data, saved predictions or models are read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[1]
SEED = 17
LIMITS = {"train": 24, "validation": 8, "test": 8}
METHODS = {
    "TF-IDF", "BM25", "Flat leaf", "Flat blend", "Flat pair ridge",
    "Primary hierarchy", "Hierarchy", "Uniform", "MLP", "LambdaMART",
    "Gate expected", "Gate mean", "Gate listnet",
}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def synthetic_data(destination):
    """Create 1,125 fictional records, 45 leaves and deliberate duplicates."""
    destination.mkdir(parents=True)
    domains = [
        ("能源", ["储能电池", "热量调节", "光伏发电"]),
        ("通信", ["无线传输", "光纤网络", "信号处理"]),
        ("制造", ["精密加工", "工业机器人", "材料成型"]),
    ]
    designs = ["闭环反馈", "多级耦合", "自适应调度", "分布协同", "容错补偿"]
    topics = [{"topic_id": "SYN_ROOT", "parent_topic_id": "", "level": 0,
               "topic_name": "纯合成演示技术树"}]
    records, assignments, by_leaf = [], [], defaultdict(list)
    for domain_i, (domain, branches) in enumerate(domains):
        domain_id = f"SYN_D{domain_i}"
        topics.append({"topic_id": domain_id, "parent_topic_id": "SYN_ROOT",
                       "level": 1, "topic_name": domain})
        for branch_i, branch in enumerate(branches):
            branch_id = f"{domain_id}_B{branch_i}"
            topics.append({"topic_id": branch_id, "parent_topic_id": domain_id,
                           "level": 2, "topic_name": branch})
            for leaf_i, design in enumerate(designs):
                leaf_id = f"{branch_id}_L{leaf_i}"
                topics.append({"topic_id": leaf_id, "parent_topic_id": branch_id,
                               "level": 3, "topic_name": f"{branch}{design}"})
                for variant in range(24):
                    serial = len(records) + 1
                    patent_id = f"SYN_{serial:06d}"
                    row = {
                        "patent_id": patent_id,
                        "family_id": f"FAM_{leaf_id}_{variant // 2:02d}",
                        "split": "synthetic_unassigned", "year": 2015 + variant % 10,
                        "title": f"合成演示{branch}{design}装置{variant:02d}",
                        "abstract": (
                            "本记录由演示程序生成，并非真实专利。"
                            f"该{domain}系统以{branch}为应用场景，通过{design}控制采样与执行。"
                            f"演示样机{serial:06d}配置{variant % 4 + 2}组传感单元，"
                            f"利用{design}计算状态误差并调节输出，测试工况为{variant:02d}。"
                        ),
                        "claims": (
                            f"一种虚构的{branch}{design}控制组件，包括状态采集模块、"
                            f"{design}决策模块和{branch}执行模块；"
                            f"采样编号{serial:06d}，窗口长度{variant + 8}，"
                            f"在{domain}实验条件下进行信号校验与状态更新。"
                        ),
                        "keywords_json": json.dumps([domain, branch, design], ensure_ascii=False),
                    }
                    records.append(row)
                    assignments.append({
                        "patent_id": patent_id, "primary_topic_id": leaf_id,
                        "secondary_topic_id": f"{branch_id}_L{(leaf_i + 1) % 5}",
                    })
                    by_leaf[leaf_id].append(patent_id)
                # Same effective input with a different family tests DSU closure.
                duplicate = dict(records[-24])
                duplicate["patent_id"] = f"SYN_{len(records) + 1:06d}"
                duplicate["family_id"] = f"FAM_DUPLICATE_{leaf_id}"
                records.append(duplicate)
                assignments.append({**assignments[-24], "patent_id": duplicate["patent_id"]})
                by_leaf[leaf_id].append(duplicate["patent_id"])
    edges = []
    for members in by_leaf.values():
        for i, patent_id in enumerate(members):
            for offset in (2, 4, 6):
                edges.append({
                    "source_patent_id": patent_id,
                    "target_patent_id": members[(i + offset) % len(members)],
                    "relation_type": "synthetic_demo_link", "split": "synthetic_unassigned",
                })
    for name, rows in [("patents", records), ("oracle_patent_topics", assignments),
                       ("oracle_topic", topics), ("graph_edges", edges)]:
        write_csv(destination / f"{name}.csv", rows)
    return {"source_records": len(records), "leaf_topics": 45,
            "deliberate_duplicate_records": 45, "generated_edges": len(edges)}


def verify_outputs(output):
    results = output / "results/revision"
    metrics = read_csv(results / f"metrics_seed{SEED}.csv")
    assert len(metrics) == len(METHODS) * 3
    assert {row["method"] for row in metrics} == METHODS
    metric_names = ["ndcg1", "ndcg5", "ndcg10", "ndcg20", "ndcg40", "mrr",
                    "precision3", "precision10", "recall10_pool",
                    "component_ci_low", "component_ci_high", "family_ci_low", "family_ci_high"]
    assert all(math.isfinite(float(row[key])) for row in metrics for key in metric_names)
    pools = read_csv(results / f"pools_seed{SEED}.csv")
    groups = defaultdict(list)
    for row in pools:
        groups[(row["partition"], row["query_id"])].append(row)
        assert row["query_component"] != row["candidate_component"]
    assert Counter(part for part, _ in groups) == Counter(LIMITS)
    assert all(len(rows) == len({row["candidate_id"] for row in rows}) == 40
               for rows in groups.values())
    scores = read_csv(results / f"pair_scores_seed{SEED}.csv")
    assert len(scores) == len(pools) == 1600
    assert all(math.isfinite(float(row[method])) for row in scores for method in METHODS)
    mapping = read_csv(results / "representative_mapping.csv")
    manifest = read_csv(results / "partition_manifest.csv")
    component_parts, family_parts = defaultdict(set), defaultdict(set)
    representative_parts = {row["patent_id"]: row["partition"] for row in manifest}
    for row in pools:
        assert representative_parts[row["query_id"]] == row["partition"]
        assert representative_parts[row["candidate_id"]] == row["partition"]
    for row in manifest:
        component_parts[row["component_id"]].add(row["partition"])
    for row in mapping:
        family_parts[row["original_family_id"]].add(representative_parts[row["representative_id"]])
    assert all(len(parts) == 1 for parts in component_parts.values())
    assert all(len(parts) == 1 for parts in family_parts.values())
    for row in read_csv(results / f"neighbors_seed{SEED}.csv"):
        for key in ("neighbor_1", "neighbor_2", "neighbor_3"):
            if row[key]:
                assert row[key] != row["patent_id"]
                assert representative_parts[row[key]] == row["partition"]
    stress = read_csv(results / f"stress50_seed{SEED}.csv")
    random = read_csv(results / f"random_metrics_seed{SEED}.csv")
    assert len(stress) == 21 and {row["noise_seed"] for row in stress} == {"17", "29", "43"}
    assert len(random) == 3
    assert all(math.isfinite(float(row["ndcg10"])) for row in stress + random)
    return {"verified": True, "methods": sorted(METHODS), "metric_rows": len(metrics),
            "ranking_pairs": len(scores), "queries": LIMITS, "candidates_per_query": 40,
            "stress_rows": len(stress), "random_baseline_rows": len(random),
            "component_and_family_partition_isolation": True,
            "finite_ranking_scores_and_summary_metrics": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "demo_output",
                        help="New or empty directory; default: repository/demo_output")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if output == REPO or output in REPO.parents:
        parser.error("The DEMO_ONLY output must be separate from the repository root.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("The output directory already exists and is not empty; choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metadata = {
        "identity": "DEMO_ONLY", "status": "running", "synthetic": True,
        "purpose": "Exercise the actual implementation; never use these scores as paper measurements.",
        "original_corpus_read": False, "split_seed": SEED, "query_limits": LIMITS,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    marker = output / "DEMO_ONLY.json"
    marker.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("DEMO_ONLY: synthetic texts and results; unrelated to the paper's measurements.", flush=True)
    try:
        metadata.update(synthetic_data(output / "data"))
        import numpy as np
        from sklearn.model_selection import GroupKFold
        from threadpoolctl import threadpool_limits
        import run_revision_experiments as experiment

        original = (experiment.ROOT, experiment.OUT, experiment.QUERY_LIMITS, sys.argv)
        try:
            experiment.ROOT, experiment.OUT = output, output / "results/revision"
            experiment.QUERY_LIMITS = dict(LIMITS)
            experiment.OUT.mkdir(parents=True)
            records, assignments, topics, _, audit = experiment.audit_data()
            assert audit["original_records"] - audit["unique_model_inputs"] == 45
            assert audit["topic_conflicts_within_identical_inputs"] == 0
            assert audit["original_families"] > audit["components"]
            partition = experiment.split_records(records, SEED)
            target = experiment.topic_targets(records, assignments, topics)
            primary = experiment.topic_targets(records, assignments, topics, primary_only=True)
            train = np.flatnonzero(partition == "train")
            components = np.asarray([row["component_id"] for row in records])
            for fit, _ in GroupKFold(3).split(train, target[train], components[train]):
                assert (target[train[fit]].sum(axis=0) > 0).all(), "Synthetic OOF label coverage"
                assert (primary[train[fit]].sum(axis=0) > 0).all(), "Synthetic primary OOF coverage"
            for part in LIMITS:
                selected = np.flatnonzero(partition == part)
                largest = max(Counter(components[selected]).values())
                assert len(selected) - largest >= 40, "Synthetic candidate coverage"
            metadata["unique_model_inputs"] = audit["unique_model_inputs"]
            metadata["components"] = audit["components"]
            metadata["all_45_leaves_in_every_oof_fit"] = True
            sys.argv = [str(Path(experiment.__file__)), "--seeds", str(SEED)]
            with threadpool_limits(limits=1):
                experiment.main()
        finally:
            experiment.ROOT, experiment.OUT, experiment.QUERY_LIMITS, sys.argv = original
        metadata.update(verify_outputs(output))
        metadata["status"] = "passed"
    except Exception as error:
        metadata.update({"status": "failed", "error": repr(error)})
        raise
    finally:
        metadata["seconds"] = round(time.perf_counter() - started, 3)
        marker.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"DEMO_ONLY passed: {metadata['metric_rows']} metric rows, 13 methods, "
          f"1,600 pairs. Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
