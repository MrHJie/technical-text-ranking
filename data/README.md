# Input data

The original research CSV files are not bundled. They were described by the authors as originating from CNIPA, but the source records do not include an independently traceable publication-number mapping or complete acquisition and redistribution terms. This repository provides the schemas, the recorded original file hashes, and a separately generated synthetic demo.

Place original UTF-8 CSV files directly in this directory. The filename and field names must match exactly. IDs are strings; empty secondary topics are permitted. The revised pipeline ignores the old `split` assignments and constructs its own component splits.

| File | Required header |
| --- | --- |
| `patents.csv` | `patent_id,family_id,split,year,title,abstract,claims,keywords_json` |
| `oracle_patent_topics.csv` | `patent_id,primary_topic_id,secondary_topic_id` |
| `oracle_topic.csv` | `topic_id,parent_topic_id,level,topic_name` |
| `graph_edges.csv` | `source_patent_id,target_patent_id,relation_type,split` |
| `retrieval_pairs.csv` | `query_patent_id,candidate_patent_id,split,oracle_relevance` |

`retrieval_pairs.csv` is required only by the older split pipeline and diagnostics. The revised experiment constructs its own candidates and grades from the other four files. Its audit hashes all CSVs present in this directory, so restoring all five is necessary to match the original five-file hash record.

## Conventions

- `patent_id` uniquely identifies a source record; the original format is `PAT_` followed by zero-padded digits. `family_id` groups related records.
- `split` uses `train`, `validation`, or `test` for the historical pipeline.
- `title`, `abstract` and `claims` are text. The revised pipeline removes everything through the first Chinese full stop `。` in the abstract; an abstract without this mark becomes empty. Do not prestrip source abstracts. `keywords_json` is preserved but is not a model input.
- `oracle_topic.csv` contains a rooted depth-three tree: one level-0 root with no parent, 5 domain nodes, 15 branch nodes, and 45 leaf nodes. Every other node points to a parent exactly one level above it.
- Each document's primary topic and optional secondary topic reference level-3 leaves. The revised target is the maximum shared ancestor depth of the documents' assigned topic sets, giving grades 0–3.
- Edge endpoints reference document IDs. The revised pipeline maps edges to input representatives, deduplicates them, excludes self/cross-partition links, and retains up to three sorted targets. The original graph-construction method is undocumented. The historical pipeline expects exactly three outgoing edges per document within its old partition.
- Historical pair rows reference query/candidate IDs; `oracle_relevance` is an integer from 0 to 3.
- A usable custom corpus needs at least 40 eligible documents outside each query's component within every partition, enough training components for three-fold cross-fitting, and sufficient positive topic examples. The original protocol's 45-topic results require the original data.

`expected_input_sha256.json` stores the five original CSV hashes. No raw record excerpts or original data are embedded in the synthetic example. Run `python scripts/run_demo.py` from the repository root to construct and use independent demo inputs.

The `.gitignore` excludes local CSV inputs and large model caches from routine Git uploads.
