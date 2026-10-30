"""Export revision figures from measured CSV/JSON results without changing inputs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'technical-text-ranking-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results/revision'
OUT = ROOT / 'figure/revision'
SEEDS = (17, 29, 43)
SEED_COLORS = {17: '#345B87', 29: '#B17D49', 43: '#568E87'}
SEED_MARKERS = {17: 'o', 29: 's', 43: '^'}
GATES = ['Gate expected', 'Gate mean', 'Gate listnet']
OLD_GATES = ['fixed_expected', 'fixed_mean', 'fixed_listnet']
GATE_LABELS = ['Expected-square', 'Mean-square', 'ListNet']
GATE_COLORS = ['#345B87', '#B17D49', '#568E87']
METHODS = ['TF-IDF', 'BM25', 'BGE-M3', 'Flat leaf', 'Flat blend', 'Flat pair ridge',
           'Primary hierarchy', 'Hierarchy', 'Uniform', 'MLP', 'LambdaMART', *GATES]
METHOD_COLORS = {'TF-IDF': '#798B9A', 'BM25': '#798B9A', 'BGE-M3': '#798B9A',
    'Flat leaf': '#A08259', 'Flat blend': '#A08259', 'Flat pair ridge': '#8E7BA8',
    'Primary hierarchy': '#5A8E89', 'Hierarchy': '#5A8E89', 'Uniform': '#A17D6D',
    'MLP': '#8E7BA8', 'LambdaMART': '#8E7BA8', **dict(zip(GATES, GATE_COLORS))}
STEMS = {1: 'rev01_protocol', 2: 'rev02_ranking', 3: 'rev03_paired',
         4: 'rev04_deletion', 5: 'rev05_corpus', 6: 'rev06_supervision',
         7: 'rev07_objectives', 8: 'rev08_gate'}
plt.rcParams.update({'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8.5,
    'axes.labelsize': 8.5, 'axes.titlesize': 9, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 8, 'svg.fonttype': 'none',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': .65, 'xtick.major.width': .65,
    'ytick.major.width': .65, 'lines.linewidth': 1.15, 'lines.markersize': 4,
    'legend.frameon': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.unicode_minus': False})


class Sources:
    def __init__(self):
        self.used = {}

    def read(self, name):
        path = SRC / name
        if path.suffix not in ('.csv', '.json'):
            raise ValueError('Only measured CSV/JSON sources are allowed')
        raw = path.read_bytes()
        self.used[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        if path.suffix == '.json':
            return json.loads(raw)
        frame = pd.read_csv(io.BytesIO(raw))
        if frame.empty:
            raise ValueError(f'{name}: empty source')
        return frame

    def verify(self):
        for name, digest in self.used.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Input changed while rendering: {name}')


def one(frame, **where):
    mask = np.ones(len(frame), dtype=bool)
    for key, val in where.items():
        mask &= frame[key].eq(val).to_numpy()
    part = frame.loc[mask]
    if len(part) != 1:
        raise ValueError(f'Expected one source row for {where}; got {len(part)}')
    return part.iloc[0]


def strict_ready(source, bge=False):
    status = source.read('pipeline_status.json')
    if status.get('prepare_only', True):
        raise RuntimeError('All strict-split runs must finish before quantitative export')
    if set(status.get('requested_seeds', [])) != set(SEEDS):
        raise RuntimeError('Expected all three requested split seeds')
    if bge:
        d = source.read('bge_metrics.csv')
        if not set(d.method).issubset({'BGE-M3', 'BGE-M3 zero-shot dense'}):
            raise ValueError('Unexpected dense baseline method')
        d = d.copy()
        d['method'] = 'BGE-M3'
        for seed in SEEDS:
            one(d, split_seed=seed, partition='test', method='BGE-M3')
        return d


def metric_frames(source):
    strict_ready(source)
    frames = [source.read(f'metrics_seed{s}.csv') for s in SEEDS]
    out = pd.concat(frames, ignore_index=True)
    for s in SEEDS:
        for m in [m for m in METHODS if m != 'BGE-M3']:
            one(out, split_seed=s, partition='test', method=m)
    return out


def panel(ax, letter, title, x=-.07):
    ax.set_title(title, loc='left', pad=9)
    ax.text(x, 1.035, letter, transform=ax.transAxes, ha='left', va='bottom',
            fontweight='bold', fontsize=10)


def grid(ax, axis='x'):
    ax.grid(axis=axis, color='#E5E9ED', lw=.55)
    ax.set_axisbelow(True)


def seed_legend(fig, y=.025, mean=True):
    hs = [Line2D([], [], color=SEED_COLORS[s], marker=SEED_MARKERS[s], lw=0,
                 label=f'Split {s}', markersize=4.5) for s in SEEDS]
    if mean:
        hs.append(Line2D([], [], color='#172D43', marker='D', lw=0,
                         label='Mean of 3 splits', markersize=4.5))
    fig.legend(handles=hs, ncol=len(hs), loc='lower center',
               bbox_to_anchor=(.5, y), columnspacing=1.4, handletextpad=.35)


def foot(fig, text, y=.02):
    fig.text(.5, y, text, ha='center', va='bottom', fontsize=8, color='#485565')


def protocol(source):
    audit = source.read('component_audit.json')
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    fig.subplots_adjust(left=.025, right=.975, bottom=.02, top=.98)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')

    def box(x, y, w, h, title, body, color='#F1F4F7', edge='#A6B2C0'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.008,rounding_size=.009',
                                  fc=color, ec=edge, lw=.8))
        ax.text(x+w/2, y+h-.037, title, weight='bold', ha='center', va='center', fontsize=8.8)
        ax.text(x+w/2, y+h/2-.025, body, ha='center', va='center', fontsize=8.1,
                linespacing=1.35)

    def arrow(a, b, dashed=False, color='#61758C', rad=0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9,
                connectionstyle=f'arc3,rad={rad}', lw=.9, color=color,
                linestyle=(0, (3, 2)) if dashed else '-'))

    ax.text(.01, .985, 'Strict component-split evaluation', fontsize=10, weight='bold', va='top')
    box(.01, .72, .285, .20, 'Normalize + deduplicate',
        f"{audit['original_records']:,} records / {audit['original_families']:,} families\nRemove abstract topic prefix\n{audit['unique_model_inputs']:,} unique model inputs")
    box(.355, .72, .285, .20, 'Transitive component union',
        f"Same family OR same exact input\nRetain family links through deduplication\n{audit['components']:,} indivisible components")
    box(.70, .72, .285, .20, 'Three independent splits',
        'Seeds 17 / 29 / 43\n70 / 15 / 15% of components\nTrain / validation / test')
    arrow((.302, .82), (.348, .82)); arrow((.647, .82), (.693, .82))
    box(.70, .405, .285, .225, 'Build 40-candidate pools',
        'Same partition, different component\n20 lexical + 20 random candidates\nQueries sampled without labels', '#EAF0F6', '#6F8CA9')
    box(.355, .405, .285, .225, 'Fit matched supervised models',
        'Flat / hierarchy / MLP / LambdaMART\nUniform + 3 gate objectives\nTrain fitting → validation selection', '#EAF0F6', '#6F8CA9')
    box(.01, .405, .285, .225, 'Evaluate frozen scores',
        'Test ranking + paired component CI\n50% query deletion × 3 noise seeds\nText + allowed links + fitted parameters', '#EAF0F6', '#6F8CA9')
    arrow((.842, .712), (.842, .638)); arrow((.692, .517), (.648, .517))
    arrow((.348, .517), (.302, .517))
    ax.text(.49, .673, 'Vocabulary, predictors and model choices are fitted without test labels',
            ha='center', va='center', fontsize=8.1, color='#485A70')
    box(.01, .035, .985-.01, .265, 'Labels are attached only AFTER candidate pools are built',
        'Ontology-derived grades → training supervision and held-out evaluation\n'
        'Candidate selection and held-out score inputs do not use oracle labels\n'
        'Semantic links: map representatives, mask cross-partition edges, cap at 3 neighbors',
        '#FAF6EF', '#B29A7B')
    arrow((.84, .398), (.84, .308), True, '#A28662')
    ax.text(.855, .351, 'attach grades', fontsize=8, va='center', color='#8B714F')
    arrow((.49, .308), (.49, .397), True, '#A28662')
    arrow((.15, .308), (.15, .397), True, '#A28662')
    ax.text(.31, .35, 'training targets', fontsize=8, va='center', color='#8B714F')
    ax.text(.01, .35, 'evaluation grades', fontsize=8, va='center', color='#8B714F')
    return fig, {'meaning': 'Label-independent pools; family/input components; three strict splits.'}


def ranking(source):
    bge = strict_ready(source, bge=True)
    metrics = metric_frames(source)
    data = pd.concat([metrics, bge], ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.25))
    fig.subplots_adjust(left=.23, right=.90, bottom=.16, top=.88)
    pos = np.arange(len(METHODS))[::-1]
    vals = []
    for y, method in zip(pos, METHODS):
        if method in GATES:
            ax.axhspan(y-.45, y+.45, color='#F0F4F8', zorder=0)
        points = [float(one(data, split_seed=s, partition='test', method=method).ndcg10) for s in SEEDS]
        vals.extend(points)
        for j, (s, val) in enumerate(zip(SEEDS, points)):
            ax.scatter(val, y+(j-1)*.16, s=21, marker=SEED_MARKERS[s],
                       facecolor=SEED_COLORS[s], edgecolor='white', lw=.35, zorder=4)
        ax.scatter(np.mean(points), y, s=27, marker='D', fc='#172D43', ec='white', lw=.45, zorder=5)
        ax.text(1.01, y, f'{np.mean(points):.4f}', transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=8)
    lo = max(0, min(vals)-.035)
    ax.set(xlim=(lo, 1.001), ylim=(-.65, len(METHODS)-.35), xlabel='Test nDCG@10',
           yticks=pos, yticklabels=METHODS)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5)); grid(ax)
    ax.text(1.01, len(METHODS)-.1, 'Mean', transform=ax.get_yaxis_transform(), fontsize=8,
            ha='left', va='bottom', weight='bold')
    fig.text(.23, .955, 'Measured ranking quality across three strict splits', fontsize=10, weight='bold')
    fig.text(.23, .914, '350 test queries and 40 candidates/query per split; BGE-M3 is frozen zero-shot.',
             fontsize=8)
    seed_legend(fig, .053)
    foot(fig, 'Points are individual split estimates; diamonds are means. No confidence interval is implied.', .015)
    return fig, {'statistics': 'Three split points and unweighted mean; no CI or SD bars.',
                 'method_count': len(METHODS)}


def paired(source):
    strict_ready(source)
    data = pd.concat([source.read(f'paired_differences_seed{s}.csv') for s in SEEDS])
    data = data[(data.partition=='test') & (data.resampling_unit=='component_id')]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
    fig.subplots_adjust(left=.19, right=.98, bottom=.20, top=.82, wspace=.22)
    relevant = data[data.comparator.isin(['Hierarchy', 'Uniform'])]
    lo = relevant['low'].min()*100; hi = relevant['high'].max()*100
    span = max(hi-lo, .001)
    xmin = min(lo-.15*span, -.001); xmax = max(hi+.18*span, .001)
    for ax, comparator, letter in zip(axes, ['Hierarchy', 'Uniform'], ['a', 'b']):
        for gi, gate in enumerate(GATES):
            for sj, seed in enumerate(SEEDS):
                y = 8-(3*gi+sj)
                r = one(data, split_seed=seed, method=gate, comparator=comparator)
                delta, low, high = np.array([r.difference, r.low, r.high], dtype=float)*100
                ax.plot([low, high], [y, y], color=SEED_COLORS[seed], lw=1.1, zorder=3)
                ax.plot([low, low], [y-.07, y+.07], color=SEED_COLORS[seed], lw=.8)
                ax.plot([high, high], [y-.07, y+.07], color=SEED_COLORS[seed], lw=.8)
                ax.scatter(delta, y, marker=SEED_MARKERS[seed], color=SEED_COLORS[seed], s=21, zorder=4)
                if delta == 0 and low == 0 and high == 0:
                    ax.annotate('0 [0, 0]', (delta, y), xytext=(7, 0), textcoords='offset points',
                                ha='left', va='center', fontsize=7.8, color='#455466')
        ax.axvline(0, lw=.8, color='#5C6976', zorder=2)
        for yy in [2.5, 5.5]: ax.axhline(yy, color='#E5E9ED', lw=.55)
        ax.set(xlim=(xmin, xmax), ylim=(-.6, 8.6), xlabel='Gate − comparator nDCG@10 (pp)')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4)); grid(ax)
        panel(ax, letter, f'Comparator: {comparator}', x=-.13)
    axes[0].set_yticks(range(8, -1, -1), [f'{label} · {s}' for label in ['Expected', 'Mean', 'ListNet'] for s in SEEDS])
    fig.text(.19, .947, 'Paired effects with component-cluster uncertainty', fontsize=10, weight='bold')
    foot(fig, 'Each row: one split, paired query-average difference and 95% whole-component bootstrap CI.', .065)
    foot(fig, 'Intervals crossing zero permit either sign. pp = percentage points; no intervals are pooled across splits.', .025)
    return fig, {'statistics': 'Per-split paired component-cluster percentile 95% CI; 1,500 draws; pp scale.'}


def deletion(source):
    metrics = metric_frames(source)
    stress = pd.concat([source.read(f'stress50_seed{s}.csv') for s in SEEDS], ignore_index=True)
    methods = ['TF-IDF', 'MLP', 'Flat pair ridge', 'Hierarchy', *GATES]
    if not np.allclose(stress.query_deletion_rate, .5):
        raise ValueError('Expected 50% deletion only')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.55), sharey=True)
    fig.subplots_adjust(left=.21, right=.98, bottom=.23, top=.82, wspace=.25)
    all_values = [[], []]
    for i, method in enumerate(methods):
        y = len(methods)-1-i
        summaries = [[], []]
        for sj, seed in enumerate(SEEDS):
            clean = float(one(metrics, split_seed=seed, partition='test', method=method).ndcg10)
            part = stress[(stress.split_seed==seed) & (stress.method==method)].sort_values('noise_seed')
            if len(part)!=3 or set(part.noise_seed)!=set(SEEDS):
                raise ValueError(f'Expected all three noise seeds: {method}, {seed}')
            corrupt = part.ndcg10.to_numpy(float)
            values = [corrupt, corrupt/clean]
            for axid, (ax, vv) in enumerate(zip(axes, values)):
                offset = (sj-1)*.20
                ax.scatter(vv, y+offset+np.array([-.045, 0, .045]), s=13,
                           marker=SEED_MARKERS[seed], color=SEED_COLORS[seed], alpha=.42,
                           edgecolor='none', zorder=3)
                ax.scatter(np.mean(vv), y+offset, s=26, marker=SEED_MARKERS[seed],
                           color=SEED_COLORS[seed], edgecolor='white', lw=.5, zorder=4)
                summaries[axid].append(np.mean(vv)); all_values[axid].extend(vv)
            axes[0].scatter(clean, y+(sj-1)*.20, s=28, marker=SEED_MARKERS[seed],
                            fc='none', ec=SEED_COLORS[seed], lw=.8, zorder=2)
            all_values[0].append(clean)
        for ax, vv in zip(axes, summaries):
            ax.scatter(np.mean(vv), y, s=27, marker='D', fc='#172D43', ec='white', lw=.5, zorder=5)
    for i, (ax, title) in enumerate(zip(axes, ['Absolute test ranking', 'Retention after deletion'])):
        lo, hi = min(all_values[i]), max(all_values[i]); span=max(hi-lo, .01)
        ax.set(xlim=(max(0,lo-.12*span), max(1.002,hi+.08*span)),
               ylim=(-.6, len(methods)-.4), xlabel='Corrupt nDCG@10' if i==0 else 'Corrupt / clean nDCG@10')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4)); grid(ax)
        panel(ax, chr(97+i), title, x=-.13)
    axes[1].axvline(1, color='#687786', lw=.8, ls=(0,(3,2)))
    axes[0].set_yticks(range(len(methods)-1,-1,-1), methods)
    fig.text(.21, .95, '50% query-character deletion on the strict splits', fontsize=10, weight='bold')
    seed_legend(fig, .09)
    foot(fig, 'Faint points: all 3 noise runs/split; solid symbols: their mean; open symbols: clean reference (a).', .05)
    foot(fig, 'Candidates, neighbor text and fitted parameters stay fixed. Nine corruption points represent three splits.', .015)
    return fig, {'statistics': '3 splits × 3 noise seeds; corrupt/clean per split; no independent-nine-run CI.'}


def corpus(source):
    metrics = metric_frames(source)
    audits = {s: source.read(f'split_audit_seed{s}.json') for s in SEEDS}
    random = pd.concat([source.read(f'random_metrics_seed{s}.csv') for s in SEEDS])
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, left=.10, right=.975, bottom=.20, top=.91,
                          height_ratios=[1.05, 1], wspace=.40, hspace=.63)
    ax = fig.add_subplot(gs[0, :]); ax.axis('off')
    rows = []; labels = []
    for s in SEEDS:
        for part in ['train', 'validation', 'test']:
            d = audits[s]['partitions'][part]
            labels.append(f'{s} · {part.title()}')
            rows.append([f"{d['records']:,}", f"{d['components']:,}",
                         f"{d['source_families']:,}", f"{d['queries']:,}", f"{d['pairs']:,}"])
    table = ax.table(cellText=rows, rowLabels=labels,
             colLabels=['Input records', 'Components', 'Source families', 'Queries', 'Pairs'],
             cellLoc='center', rowLoc='left', loc='center', bbox=[.09, -.05, .91, 1.])
    table.auto_set_font_size(False); table.set_fontsize(8)
    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor('#E1E6EB'); cell.set_linewidth(.45)
        if r==0: cell.set_facecolor('#EAF0F6'); cell.set_text_props(weight='bold')
        elif (r-1)//3%2==0: cell.set_facecolor('#F7F9FB')
    panel(ax, 'a', 'Actual partition sizes after component assignment', x=-.04)
    ax = fig.add_subplot(gs[1, 0])
    bottoms = np.zeros(3)
    palette = ['#E0E5EA', '#B3C2D0', '#7B99B4', '#3F648B']
    for grade, color in enumerate(palette):
        values = np.array([audits[s]['partitions']['test']['grade_counts'].get(str(grade),0) for s in SEEDS], float)
        totals = np.array([audits[s]['partitions']['test']['pairs'] for s in SEEDS],float)
        vals = values/totals*100
        ax.bar(range(3), vals, width=.63, bottom=bottoms, color=color, edgecolor='white', lw=.5, label=f'Grade {grade}')
        for x, b, v in zip(range(3), bottoms, vals):
            if v >= 6:
                ax.text(x, b+v/2, f'{v:.1f}', ha='center', va='center', fontsize=8,
                        color='white' if grade>=2 else '#233649')
        bottoms += vals
    ax.set(xticks=range(3), xticklabels=[str(s) for s in SEEDS], ylim=(0,100),
           xlabel='Split seed', ylabel='Test-pool pair grades (%)')
    panel(ax, 'b', 'Observed grade mix', x=-.18)
    ax.legend(ncol=2, fontsize=8, loc='lower center', bbox_to_anchor=(.5,-.44), columnspacing=.7)
    ax = fig.add_subplot(gs[1, 1])
    for pj, part in enumerate(['train','validation','test']):
        for sj, s in enumerate(SEEDS):
            y = 2-pj+(sj-1)*.14
            rv = float(one(random, split_seed=s, partition=part).ndcg10)
            tv = float(one(metrics, split_seed=s, partition=part, method='TF-IDF').ndcg10)
            ax.plot([rv,tv],[y,y],color=SEED_COLORS[s],alpha=.5,lw=.8)
            ax.scatter(rv,y,s=22,marker=SEED_MARKERS[s],fc='white',ec=SEED_COLORS[s],lw=.8,zorder=4)
            ax.scatter(tv,y,s=22,marker=SEED_MARKERS[s],color=SEED_COLORS[s],lw=0,zorder=4)
    ax.set(yticks=[2,1,0], yticklabels=['Train','Validation','Test'], ylim=(-.45,2.45),
           xlim=(0,1.025), xlabel='nDCG@10 on actual candidate pools')
    ax.set_xticks([0,.5,1]); grid(ax)
    panel(ax, 'c', 'Exact random expectation → TF-IDF', x=-.18)
    seed_legend(fig, .017, mean=False)
    fig.text(.51,.065,'c: open = exact random-permutation expectation; filled = TF-IDF.',ha='center',fontsize=8)
    return fig, {'statistics': 'Actual partition counts and pair grades; exact random permutation expectation.'}


def supervision(source):
    metrics = metric_frames(source)
    methods = ['Flat leaf','Flat blend','Flat pair ridge','Hierarchy','Primary hierarchy']
    fig, axes = plt.subplots(1,2,figsize=(7.2,3.95),sharey=True)
    fig.subplots_adjust(left=.22,right=.975,bottom=.25,top=.78,wspace=.24)
    for ax,metric,letter,title in zip(axes,['ndcg10','ndcg40'],['a','b'],
            ['Top-10 ranking','Full-pool ordering (secondary)']):
        allvals=[]
        for y,m in zip(range(4,-1,-1),methods):
            vals=[float(one(metrics,split_seed=seed,partition='test',method=m)[metric]) for seed in SEEDS]
            allvals.extend(vals)
            for j,(seed,v) in enumerate(zip(SEEDS,vals)):
                ax.scatter(v,y+(j-1)*.16,s=23,marker=SEED_MARKERS[seed],color=SEED_COLORS[seed],zorder=4)
            ax.scatter(np.mean(vals),y,s=26,marker='D',fc='#172D43',ec='white',lw=.5,zorder=5)
        lo=max(0,min(allvals)-.003)
        ax.set(ylim=(-.65,4.65),xlim=(lo,1.0005),xlabel='Test nDCG@10' if metric=='ndcg10' else 'Test nDCG@40')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4));grid(ax);panel(ax,letter,title,x=-.14)
    axes[0].set_yticks(range(4,-1,-1),methods)
    fig.text(.22,.94,'Matched supervision and multibranch topic controls',fontsize=10,weight='bold')
    fig.text(.22,.88,'Same three strict splits; mixing weights are independently selected on validation.',fontsize=8)
    seed_legend(fig,.075)
    foot(fig,'Each point is one strict split; diamonds are means. No confidence interval is implied.',.022)
    return fig, {'statistics': 'Three split points and means for nDCG@10 and secondary nDCG@40; no CI.'}


def objectives(source):
    metrics=source.read('gate_fixed_objective_metrics.csv')
    stats=source.read('gate_fixed_objective_statistics.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.9),sharey=True,gridspec_kw={'width_ratios':[1.15,1,1.2]})
    fig.subplots_adjust(left=.19,right=.975,bottom=.27,top=.72,wspace=.30)
    for i,(m,label,c) in enumerate(zip(OLD_GATES,GATE_LABELS,GATE_COLORS)):
        y=2-i
        for j,(p,mark) in enumerate([('original','o'),('text_disjoint','^')]):
            mr=one(metrics,method=m,protocol=p);st=one(stats,method=m,protocol=p)
            vals=[float(mr.ndcg10),float(st.expected_edges_mean),float(st.mask_variance_mean)]
            for k,(ax,v) in enumerate(zip(axes,vals)):
                yy=y+(.12 if j==0 else -.12)
                ax.scatter(v,yy,s=28,marker=mark,fc=c if j==0 else 'white',ec=c,lw=.85,zorder=4)
                if k==2:
                    ax.annotate(f'{v:.1e}',(v,yy),xytext=(0,7 if j==0 else -9),
                                textcoords='offset points',ha='center',va='center',fontsize=7.6,color=c)
    axes[0].set(yticks=[2,1,0],yticklabels=GATE_LABELS,ylim=(-.45,2.55),
                xlim=(.9910,.9972),xlabel='Test nDCG@10')
    axes[0].set_xticks([.992,.994,.996]);axes[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    axes[1].set(xlim=(0,3),xlabel='Mean expected edges')
    axes[1].set_xticks([0,1,2,3])
    axes[2].set(xscale='log',xlim=(3e-19,2e-8),xlabel='Mean mask-induced score variance')
    axes[2].set_xticks([1e-17,1e-13,1e-9]);axes[2].xaxis.set_minor_locator(ticker.NullLocator())
    for ax,l,t in zip(axes,['a','b','c'],['Ranking quality','Edge density','Mask variance']):
        grid(ax);panel(ax,l,t,x=-.15)
    fig.text(.19,.945,'Old-split mechanism diagnostic',fontsize=10,weight='bold')
    fig.text(.19,.875,'Fixed α = 0.5, ρ = 0.3, prior = 0.25, β = 0.002; no joint α/ρ retuning.',fontsize=8.5)
    hs=[Line2D([],[],marker='o',color='#62798F',lw=0,label='Old original test (266 queries)',ms=5),
        Line2D([],[],marker='^',color='#62798F',mfc='white',lw=0,label='Old text-filtered subset (149 queries)',ms=5)]
    fig.legend(handles=hs,ncol=1,loc='lower center',bbox_to_anchor=(.55,.075))
    foot(fig,'Points are fixed-fit measurements; tiny mask variance does not establish calibrated uncertainty.',.015)
    return fig, {'protocol': 'OLD SPLIT mechanism only; original and text-filtered subset.',
                 'parameters': {'alpha':.5,'rho':.3,'prior':.25,'beta':.002}}


def gate(source):
    fits=source.read('gate_fixed_objective_fits.json')
    stats=source.read('gate_fixed_objective_statistics.csv')
    sensitivity=source.read('gate_sensitivity_metrics.csv')
    sens_stats=source.read('gate_sensitivity_statistics.csv')
    fig=plt.figure(figsize=(7.2,7.25))
    ax=fig.add_axes([.19,.715,.73,.18])
    theta=np.array([fits[k]['theta'] for k in ['expected','mean','listnet']],float)
    limit=np.max(np.abs(theta))
    im=ax.pcolormesh(np.arange(7)-.5,np.arange(4)-.5,theta,cmap='RdBu_r',norm=colors.SymLogNorm(linthresh=100,linscale=.8,vmin=-limit,vmax=limit),shading='flat')
    ax.set(xlim=(-.5,5.5),ylim=(2.5,-.5))
    for i in range(3):
        for j in range(6):
            ax.text(j,i,f'{theta[i,j]:,.0f}',ha='center',va='center',fontsize=8,
                    color='white' if abs(theta[i,j])>2500 else '#213446')
    ax.set_xticks(range(6),['Bias','q–n cosine','d–n cosine','|h(q,n)−h(q,d)|','d entropy','n entropy'])
    ax.tick_params(axis='x',length=0,pad=5,labelsize=7.7)
    ax.set_yticks(range(3),GATE_LABELS);ax.tick_params(axis='y',length=0)
    panel(ax,'a','Raw gate coefficients (symmetric-log color scale)',x=-.16)
    cax=fig.add_axes([.936,.715,.017,.18]);cb=fig.colorbar(im,cax=cax,ticks=[-10000,0,10000]);cb.ax.tick_params(labelsize=7);cb.solids.set_rasterized(False)
    fig.text(.19,.662,'q: query; d: candidate; n: neighbor. Features standardized except bias.',fontsize=8)
    ax=fig.add_axes([.19,.455,.37,.135]);axb=fig.add_axes([.64,.455,.29,.135],sharey=ax)
    for i,(m,c) in enumerate(zip(OLD_GATES,GATE_COLORS)):
        y=2-i; r=one(stats,method=m,protocol='original')
        ax.plot([r.gate_q05,r.gate_q95],[y,y],lw=1.2,color=c)
        ax.plot([r.gate_q05,r.gate_q95],[y,y],lw=0,marker='|',ms=7,color=c)
        ax.scatter(r.gate_q50,y,s=27,color=c,ec='white',lw=.5,zorder=4)
        low=float(r.saturation_below_001)*100;high=float(r.saturation_above_099)*100
        soft=max(0,100-low-high)
        axb.barh(y,low,height=.54,color='#7B96AE')
        axb.barh(y,soft,left=low,height=.54,color='#E7CF9F')
        axb.barh(y,high,left=low+soft,height=.54,color='#405F87')
        axb.text(50,y,f'{low:.2f}% / {high:.2f}%',ha='center',va='center',fontsize=8,color='white')
    ax.set(xlim=(-.025,1.025),ylim=(-.5,2.5),yticks=[2,1,0],yticklabels=GATE_LABELS,
           xlabel='Gate probability: 5th / 50th / 95th percentiles')
    ax.set_xticks([0,.5,1]);grid(ax)
    panel(ax,'b','Probability distribution: old original test',x=-.31)
    axb.set(xlim=(0,100),xlabel='Share: p < .01 / p > .99')
    axb.set_xticks([0,50,100]);axb.tick_params(axis='y',labelleft=False,left=False)
    fig.text(.19,.367,'Middle (gold): .01 ≤ p ≤ .99; percentile bars are distribution summaries, not CIs.',fontsize=8)
    betas=[0,.0002,.002,.02];priors=[.1,.25,.5]
    vmin=float(sensitivity.ndcg10.min());vmax=float(sensitivity.ndcg10.max())
    for ix,(p,title) in enumerate([('original','Old original test'),('text_disjoint','Old text-filtered subset')]):
        ax=fig.add_axes([.19+ix*.42,.135,.33,.18])
        mat=np.array([[float(one(sensitivity,protocol=p,prior=prior,beta=beta).ndcg10)
                       for beta in betas] for prior in priors])
        ax.pcolormesh(np.arange(5)-.5,np.arange(4)-.5,mat,cmap='Blues',vmin=vmin,vmax=vmax,shading='flat')
        ax.set(xlim=(-.5,3.5),ylim=(2.5,-.5))
        for i in range(3):
            for j in range(4):
                frac=(mat[i,j]-vmin)/max(vmax-vmin,1e-12)
                ax.text(j,i,f'{mat[i,j]:.6f}',ha='center',va='center',fontsize=7.8,
                        color='white' if frac>.62 else '#20364C')
        ax.set_xticks(range(4),['0','0.0002','0.002','0.02']);ax.set_yticks(range(3),['0.10','0.25','0.50'])
        ax.set(xlabel='KL coefficient β',ylabel='Gate prior');ax.tick_params(length=0)
        panel(ax,chr(99+ix),title,x=-.22)
    fig.text(.19,.975,'Old-split gate diagnostics and full sensitivity grid',fontsize=10,weight='bold')
    fig.text(.19,.944,'Fixed α = 0.5, ρ = 0.3. Sensitivity uses mean-square fitting; every prior × β setting is shown.',fontsize=8)
    soft=one(sens_stats,protocol='original',prior=.25,beta=.02)
    pct=(1-float(soft.saturation_below_001)-float(soft.saturation_above_099))*100
    foot(fig,f'At prior .25, β = .02: {pct:.2f}% of gates are inside [.01,.99]; c,d annotate test nDCG@10.',.025)
    return fig, {'protocol':'OLD SPLIT mechanism only; no strict-split evidence.',
                 'statistics':'Raw theta; p quantiles and saturation; all 12 mean-square sensitivity settings per old protocol.'}


DRAW={1:protocol,2:ranking,3:paired,4:deletion,5:corpus,6:supervision,7:objectives,8:gate}


def export_one(number, manifest):
    source=Sources()
    fig,metadata=DRAW[number](source)
    if not np.isclose(fig.get_size_inches()[0],7.2):
        raise ValueError('Figure width must be exactly 7.2 inches')
    OUT.mkdir(parents=True,exist_ok=True)
    stem=OUT/STEMS[number]
    for ext in ['svg','pdf','png']:
        fig.savefig(stem.with_suffix('.'+ext),dpi=150,facecolor='white')
    plt.close(fig)
    source.verify()
    manifest['figures'][STEMS[number]]={
        'files':[str(stem.with_suffix('.'+ext).relative_to(ROOT)) for ext in ['svg','pdf','png']],
        'source_sha256':source.used, **metadata}
    print(f'Exported {STEMS[number]}',flush=True)


def contact_sheet():
    available=[n for n in range(1,9) if (OUT/(STEMS[n]+'.png')).exists()]
    if not available:return
    rows=(len(available)+1)//2
    fig,axes=plt.subplots(rows,2,figsize=(14.4,rows*5.1),squeeze=False)
    fig.subplots_adjust(left=.015,right=.985,bottom=.01,top=.99,hspace=.12,wspace=.035)
    for ax,n in zip(axes.ravel(),available):
        ax.imshow(plt.imread(OUT/(STEMS[n]+'.png')));ax.axis('off')
        ax.set_title(STEMS[n],loc='left',fontsize=11,pad=4)
    for ax in axes.ravel()[len(available):]:ax.axis('off')
    fig.savefig(OUT/'contact_sheet.png',dpi=110,facecolor='white');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',nargs='+',type=int,choices=range(1,9),default=list(range(1,9)))
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'source_manifest.json'
    manifest=json.loads(target.read_text()) if target.exists() else {'figures':{}}
    manifest.update({'backend':'Python/matplotlib','figure_width_inches':7.2,'png_dpi':150,
        'svg_text_editable':True,'pdf_fonttype':42,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_policy':'Read-only measured CSV/JSON; each figure input hash verified after export.'})
    for number in args.figures:export_one(number,manifest)
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    contact_sheet()
    print('Source hashes unchanged; source_manifest.json and contact_sheet.png updated.',flush=True)


if __name__=='__main__':main()
