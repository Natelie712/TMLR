"""CPU/public-text-only Phase A audit; pins come from public_metadata.json.

Raw downloads stay in /tmp. No credentials, model loading or GPU work.
Canonical IDs group whitespace duplicates and literal A/B swaps, not paraphrases.
"""
import concurrent.futures
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = Path('/tmp/curveball_table1_data_20261022')
CONCEPTS = ['self_awareness_general', 'wealth_seeking', 'power_seeking',
            'corrigible_more', 'humor', 'rude', 'excitement', 'sadness']


def digest(b):
    return hashlib.sha256(b).hexdigest()


def norm(s):
    return ' '.join(str(s).split())


def group_id(prompt, behavior):
    if behavior:
        marks = list(re.finditer(r'\([AB]\)', prompt))
        if len(marks) < 2 or [m.group() for m in marks[-2:]] != ['(A)', '(B)']:
            return None
        a, b = marks[-2:]
        value = [norm(prompt[:a.start()]),
                 *sorted([norm(prompt[a.end():b.start()]), norm(prompt[b.end():])])]
    else:
        value = norm(prompt)
    return digest(json.dumps(value, ensure_ascii=True).encode())


def fetch(repo, revision):
    p = CACHE / (repo.replace('/', '--') + '--' + revision + '.parquet')
    if not p.exists():
        req = Request('https://huggingface.co/datasets/' + repo + '/resolve/' + revision
                      + '/data/train-00000-of-00001.parquet',
                      headers={'User-Agent': 'curveball-phase-a-public-audit'})
        with urlopen(req, timeout=120) as r:
            p.write_bytes(r.read())
    return p


def audit(item):
    concept, meta = item
    p = fetch(meta['repo'], meta['revision'])
    rows = pq.read_table(p).to_pylist()
    behavior = concept in CONCEPTS[:4]
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[group_id(row['prompt'], behavior)].append((i, row))
    localpath = ROOT / 'automated_datasets' / concept / (concept + '-eval.json')
    local = json.loads(localpath.read_text())
    examples = local['examples']
    local_ids = [group_id(x['prompt'], behavior) for x in examples]
    train_ids = set(groups) - {None}
    overlap = train_ids.intersection(x for x in local_ids if x is not None)
    label_patterns = Counter(','.join(map(str, sorted(x['label'] for _, x in rs)))
                             for gid, rs in groups.items() if gid is not None)
    def strict_bad(prompt):
        marks = list(re.finditer(r'\([AB]\)', prompt))
        if [m.group() for m in marks] != ['(A)', '(B)']:
            return True
        a, b = marks
        choices = [norm(prompt[a.end():b.start()]), norm(prompt[b.end():])]
        return not all(choices) or choices[0] == choices[1]

    conflicts = []
    if behavior:
        for gid, rs in groups.items():
            targets = set()
            if gid is None:
                continue
            for _, row in rs:
                if row['label'] != 1 or row['response'].strip() not in ('(A)', '(B)'):
                    continue
                marks = list(re.finditer(r'\([AB]\)', row['prompt']))
                a, b = marks[-2:]
                targets.add(norm(row['prompt'][a.end():b.start()] if row['response'].strip() == '(A)'
                                 else row['prompt'][b.end():]))
            if len(targets) > 1:
                conflicts.append(gid)
    result = {
        'repo': meta['repo'], 'revision': meta['revision'],
        'parquet_sha256': digest(p.read_bytes()), 'parquet_bytes': p.stat().st_size,
        'rows': len(rows), 'columns': list(rows[0]),
        'labels': dict(Counter(str(x['label']) for x in rows)),
        'unique_prompt_strings': len({x['prompt'] for x in rows}),
        'canonical_prompt_groups': len(train_ids),
        'invalid_ab_rows': len(groups.get(None, [])) if behavior else 0,
        'strict_invalid_choice_rows': sum(strict_bad(x['prompt']) for x in rows) if behavior else None,
        'invalid_answer_response_rows': sum(x['response'].strip() not in ('(A)', '(B)') for x in rows) if behavior else None,
        'conflicting_positive_answer_groups': conflicts if behavior else None,
        'blank_prompt_rows': sum(not norm(x['prompt']) for x in rows),
        'blank_response_rows': sum(not norm(x['response']) for x in rows),
        'duplicate_full_rows': len(rows) - len({json.dumps(x, sort_keys=True) for x in rows}),
        'group_label_patterns': dict(label_patterns),
        'local_eval': str(localpath.relative_to(ROOT)),
        'local_eval_sha256': digest(localpath.read_bytes()),
        'local_declared_total': local.get('total_examples'), 'local_actual_rows': len(examples),
        'local_unique_groups': len(set(local_ids) - {None}),
        'local_invalid_ab_rows': sum(x is None for x in local_ids) if behavior else 0,
        'local_strict_invalid_choice_rows': sum(strict_bad(x['prompt']) for x in examples) if behavior else None,
        'local_blank_prompts': sum(not norm(x['prompt']) for x in examples),
        'train_local_eval_overlap_groups': len(overlap),
        'train_local_eval_overlap_rows': sum(x in overlap for x in local_ids),
        'local_bad_answer_mappings': sum({x.get('matching_answer'), x.get('non_matching_answer')}
                                        != {'(A)', '(B)'} for x in examples) if behavior else None,
        'sample_pair': [{k: v for k, v in row.items()} for _, row in next(iter(groups.values()))][:2],
    }
    manifest = [{'group_id': gid, 'source_row_indices': [i for i, _ in rs],
                 'seen_in_local_eval': gid in overlap} for gid, rs in sorted(
                     groups.items(), key=lambda kv: str(kv[0]))]
    (HERE / (concept + '_source_groups.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    return concept, result


def main():
    CACHE.mkdir(exist_ok=True)
    metadata = json.loads((HERE / 'public_metadata.json').read_text())['datasets']
    items = [(c, metadata['shiv96/convsersations_' + c + '_large']) for c in CONCEPTS]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for c, r in pool.map(audit, items):
            results[c] = r
            print(c, r['rows'], r['canonical_prompt_groups'], r['local_actual_rows'],
                  'overlap', r['train_local_eval_overlap_groups'], flush=True)
    mirror = metadata['curveball-steering/conversations_humor_large']
    p = fetch(mirror['repo'], mirror['revision'])
    original = fetch(items[4][1]['repo'], items[4][1]['revision'])
    output = {'canonicalization': 'whitespace collapse; sort literal final A/B option texts; SHA256 JSON; no semantic deduplication',
              'datasets': results, 'humor_mirror': {
                  'repo': mirror['repo'], 'revision': mirror['revision'],
                  'parquet_sha256': digest(p.read_bytes()),
                  'byte_identical_to_shiv96': p.read_bytes() == original.read_bytes(),
                  'rows_identical_to_shiv96': pq.read_table(p).equals(pq.read_table(original))}}
    (HERE / 'data_audit.json').write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
