"""Generate manuscript numbers and tables from the completed fixed revision runs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ACCESS_latex_template_20260513'
R=ROOT/'results/revision'
SEEDS=[17,29,43]
METHODS=['TF-IDF','BM25','BGE-M3 zero-shot dense','MLP','LambdaMART','Flat leaf','Flat blend','Flat pair ridge','Primary hierarchy','Hierarchy','Uniform','Gate expected','Gate mean','Gate listnet']
LABEL={'BGE-M3 zero-shot dense':'BGE-M3 dense','Flat pair ridge':'Flat pair ridge','Primary hierarchy':'Primary-only hierarchy','Hierarchy':'Hierarchical blend','Uniform':'Uniform neighbors','Gate expected':'Expected-mask gate','Gate mean':'Mean-score gate','Gate listnet':'Listwise gate'}
def name(s):return LABEL.get(s,s)
def macro(key,val):return '\\newcommand{\\Rev'+key+'}{'+val+'}\n'
def main():
 status=json.loads((R/'pipeline_status.json').read_text());assert status['prepare_only'] is False
 frames=[pd.read_csv(R/f'metrics_seed{s}.csv') for s in SEEDS]
 frame=pd.concat(frames);test=frame[frame.partition=='test'].copy()
 bge=pd.read_csv(R/'bge_metrics.csv');test=pd.concat([test,bge[bge.partition=='test']],ignore_index=True)
 assert len(test)==42
 means=test.groupby('method').ndcg10.mean()
 stress=pd.concat([pd.read_csv(R/f'stress50_seed{s}.csv') for s in SEEDS]);stress['clean_ndcg10']=[float(test[(test.split_seed==r.split_seed)&(test.method==r.method)].iloc[0].ndcg10) for r in stress.itertuples()]
 stress['retention']=stress.ndcg10/stress.clean_ndcg10
 sm=stress.groupby('method')[['ndcg10','retention']].mean()
 macrotext='% Generated from completed revision results. Do not type performance values here.\n'
 for k,m in [('HierarchyMean','Hierarchy'),('FlatMean','Flat leaf'),('TfidfMean','TF-IDF'),('BmMean','BM25'),('BgeMean','BGE-M3 zero-shot dense'),('MlpMean','MLP'),('RankerMean','LambdaMART')]:macrotext+=macro(k,f'{means[m]:.4f}')
 for k,m in [('HierarchyNoisy','Hierarchy'),('TfidfNoisy','TF-IDF'),('MlpNoisy','MLP')]:macrotext+=macro(k,f'{sm.loc[m,"ndcg10"]:.4f}')
 for k,m in [('HierarchyRetention','Hierarchy'),('TfidfRetention','TF-IDF'),('MlpRetention','MLP')]:macrotext+=macro(k,f'{100*sm.loc[m,"retention"]:.2f}\\%')
 (OUT/'revision_numbers.tex').write_text(macrotext)
 # Actual partition and annotation counts.
 corpus=r'''\begin{table*}[t]
\caption{Actual document partition sizes after deduplication and transitive grouping. Every split has 1,200/250/350 training/validation/test queries, with forty candidates per query. Each partition covers all forty-five leaves.}
\label{tab:partitions}\centering\small
\begin{tabular}{lrrrrrr}\toprule
Split seed & Train records & Validation records & Test records & Train components & Validation components & Test components\\\midrule
'''
 for seed in SEEDS:
  audit=json.loads((R/f'split_audit_seed{seed}.json').read_text())['partitions']
  corpus+=str(seed)+' & '+' & '.join(f'{audit[p][k]:,}' for k in ['records','components'] for p in ['train','validation','test'])+r'\\'+'\n'
 corpus+=r'\bottomrule\end{tabular}\end{table*}'+'\n'
 performance=r'''\begin{table*}[t]
\caption{Held-out nDCG@10 on three fixed component-disjoint partitions. Means and sample standard deviations describe split variation, not a confidence interval across independent datasets. Topic-supervised methods and text-only methods have different information sets; flat-versus-hierarchical comparisons match topic supervision.}
\label{tab:performance}\centering\small
\begin{tabular}{lrrrrr}\toprule
Score & Seed 17 & Seed 29 & Seed 43 & Mean & Split SD\\\midrule
'''
 for m in METHODS:
  v=test[test.method==m].sort_values('split_seed').ndcg10.to_numpy();assert len(v)==3
  performance+=name(m)+' & '+' & '.join(f'{x:.6f}' for x in [*v,v.mean(),v.std(ddof=1)])+r'\\'+'\n'
 performance+=r'\bottomrule\end{tabular}\end{table*}'+'\n'
 full=r'''\begin{table}[t]
\caption{Secondary full-pool ordering and multi-branch control. Values are mean nDCG@40 across the three fixed splits. This cutoff is secondary to nDCG@10.}
\label{tab:fullpool}\centering\small
\begin{tabular}{lr}\toprule
Score & Mean nDCG@40\\\midrule
'''
 for m in ['Flat leaf','Flat blend','Flat pair ridge','Primary hierarchy','Hierarchy','Uniform','Gate expected','Gate mean','Gate listnet']:
  full+=name(m)+f' & {test[test.method==m].ndcg40.mean():.6f}'+r'\\'+'\n'
 full+=r'\bottomrule\end{tabular}\end{table}'+'\n'
 selection=r'''\begin{table*}[t]
\caption{Validation-selected hierarchy and graph coefficients. The reported edge expectation averages permitted probabilities over all test pairs, including pairs without an available neighbor. The mean permitted-edge count is reported separately in the graph audit.}
\label{tab:selection}\centering\small
\begin{tabular}{llrrrr}\toprule
Split & Score & $\alpha$ & $\rho$ & Expected edges / pair & Mean mask variance\\\midrule
'''
 for seed in SEEDS:
  s=json.loads((R/f'selection_seed{seed}.json').read_text())['selection']
  for m in ['Hierarchy','Uniform','Gate expected','Gate mean','Gate listnet']:
   v=s[m];selection+=f'{seed} & {name(m)} & {v["alpha"]:.2f} & {v["rho"]:.2f} & '
   selection+=(f'{v["test_expected_edges"]:.4f} & {v["test_mask_variance"]:.2e}' if 'test_expected_edges' in v else '--- & ---')+r'\\'+'\n'
 selection+=r'\bottomrule\end{tabular}\end{table*}'+'\n'
 (OUT/'revision_tables.tex').write_text(corpus+performance+full+selection)
 stress.to_csv(R/'stress50_with_retention.csv',index=False)
 pd.DataFrame({'method':means.index,'mean_ndcg10':means.values}).to_csv(R/'manuscript_metric_means.csv',index=False)
 print('Generated revision_numbers.tex and revision_tables.tex; 42 primary scores and 63 perturbation rows.')
if __name__=='__main__':main()
