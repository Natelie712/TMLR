"""Standard-library verification of the curated public artifact release."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'TMLR_paper'
MODELS={'qwen3','llama3b','llama2'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    manifest=json.loads((ROOT/'release_manifest.json').read_text())
    for f in manifest['files']:
        path=ROOT/f['destination']
        assert path.is_file() and not path.is_symlink(),path
        assert sha(path)==f['packaged_sha256'],path
        assert path.stat().st_size<95*1024**2,path
    for path in ROOT.rglob('*'):
        if '.git' in path.parts or 'reproduced' in path.parts:
            continue
        assert not path.is_symlink(),path
        assert not any(s.lower() in ['gemma','phi'] for s in path.relative_to(ROOT).parts),path
    mainrows=rows(P/'data/main_results.csv')
    assert len(mainrows)==21 and {r['model'] for r in mainrows}==MODELS
    selected=json.loads((ROOT/'results/small_lambda_validation_test_20261110/frozen_config.json').read_text())['selected']
    selections={(r['model'],r['concept']):float(r['selected_weight']) for r in selected}
    llama=json.loads((ROOT/'results/llama2_small_lambda_positive_20261111/selected_configurations.json').read_text())['selected']
    selections.update({('llama2',c):float(w) for c,w in llama.items()})
    for r in mainrows:
        assert int(r['questions'])==50
        assert abs(float(r['curveball_delta_pp'])-float(r['linear_delta_pp'])-float(r['difference_pp']))<1e-8
        assert float(r['selected_lambda'])==selections[r['model'],r['concept']]
        assert abs(100*(float(r['positive_probability'])-float(r['nohook_probability']))-float(r['curveball_delta_pp']))<1e-8
    counts=rows(P/'data/dataset_statistics.csv')
    assert sum(int(r['fit_questions']) for r in counts)==3906
    assert sum(int(r['validation_questions']) for r in counts)==912
    assert sum(int(r['benchmark_questions']) for r in counts)==350
    assert json.loads((P/'data/negative_lambda_search/completion.json').read_text())['complete']
    negative=rows(P/'data/negative_lambda_search/summary.csv')
    assert {r['model'] for r in negative}==MODELS
    negselections={m:json.loads((P/f'data/negative_lambda_search/selected_{m}.json').read_text())['selected'] for m in MODELS}
    for r in negative:
        assert abs(100*(float(r['negative_probability'])-float(r['nohook_probability']))-float(r['negative_delta_pp']))<1e-8
        assert float(r['selected_lambda'])==float(negselections[r['model']][r['concept']])
    # Recompute every positive main-table cell directly from released question records.
    per={}
    for rel in ['results/small_lambda_validation_test_20261110/test_per_question.csv.gz',
                'results/llama2_small_lambda_positive_20261111/per_question.csv.gz']:
        with gzip.open(ROOT/rel,'rt',newline='') as f:
            for r in csv.DictReader(f):
                assert r['model'] in MODELS
                if r.get('partition','benchmark')!='benchmark':
                    continue
                key=r['model'],r['concept'],r['method']
                per.setdefault(key,[]).append((int(r['index']),100*(float(r['probability'])-float(r['nohook_probability']))))
    for r in mainrows:
        for method,field in [('Linear','linear_delta_pp'),('lambda_'+str(float(r['selected_lambda'])),'curveball_delta_pp')]:
            values=per[r['model'],r['concept'],method]
            assert len(values)==len({i for i,v in values})==50
            assert abs(statistics.mean(v for i,v in values)-float(r[field]))<1e-8,(r,method)
    with gzip.open(P/'data/negative_lambda_search/per_question.csv.gz','rt',newline='') as f:
        neg={}
        for r in csv.DictReader(f):
            assert r['model'] in MODELS
            if r['partition']=='benchmark' and r.get('negative_delta_pp'):
                neg.setdefault((r['model'],r['concept'],r['method']),[]).append((r['index'],float(r['negative_delta_pp'])))
    for r in negative:
        if r['partition']!='benchmark':
            continue
        values=neg[r['model'],r['concept'],r['method']]
        assert len(values)==len({i for i,v in values})==50
        assert abs(statistics.mean(v for i,v in values)-float(r['negative_delta_pp']))<1e-8
    for path in (P/'data').rglob('*.csv'):
        for r in rows(path):
            if r.get('model'):
                assert r['model'] in MODELS,(path,r['model'])
    assert len(list((P/'figures').glob('figure_*.pdf')))==18
    assert not (P/'figures/drafts').exists()
    print('PASS: packaged hashes; 21 positive cells and all negative test means reproduced from question records; three models only; split counts; frozen selections; 18 completed figures.')


if __name__=='__main__':
    main()
