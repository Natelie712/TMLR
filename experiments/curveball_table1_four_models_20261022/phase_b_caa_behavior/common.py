"""Pinned prompts and common CAA-compatible decision boundary."""
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'experiments/caa_reproduction')]
from linear_eval import chat_tokens, instruction_boundary, score_logits, translation_hook
from src.mcq_symmetrization import symmetrize_questions
CONFIG = json.loads((HERE/'config.json').read_text())
AUDIT = json.loads((HERE.parent/'local_model_audit.json').read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(key):
    return AUDIT[CONFIG['models'][key]['repo']]['local_snapshot'] or '/tmp/phase_b_qwen3_20261022'


def tokenizer(key, path=None):
    return AutoTokenizer.from_pretrained(path or snapshot(key), local_files_only=True, token=False)


def prompt(tok, key, question, answer='('):
    if key == 'llama2':
        ids = chat_tokens(tok, question, answer)
        boundary = instruction_boundary(ids, tok)
        rendered = f'[INST] {question.strip()} [/INST] {answer.strip()}'
    else:
        rendered = tok.apply_chat_template([{'role':'user', 'content':question.strip()}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    date_string=CONFIG['prompt']['llama_date_string']) + answer
        ids = tok.encode(rendered, add_special_tokens=False)
        marker = {'llama3b':'<|eot_id|>', 'phi':'<|end|>', 'qwen3':'<|im_end|>'}[key]
        marker_id = tok.convert_tokens_to_ids(marker)
        positions = [i for i, v in enumerate(ids) if v == marker_id]
        if not positions:
            raise ValueError('Missing final user delimiter')
        boundary = positions[-1]
    return {'input_ids':ids, 'boundary':boundary, 'rendered':rendered}


def decision(tok, key, question):
    prefix = prompt(tok,key,question)
    answer_ids, completions, reencoded_stable = {}, {}, {}
    for letter in ('A','B'):
        reencoded = prompt(tok,key,question,'('+letter+')')['input_ids']
        n = len(prefix['input_ids'])
        reencoded_stable[letter] = reencoded[:n] == prefix['input_ids']
        if key in ('llama2', 'phi'):
            complete = reencoded
            assert reencoded_stable[letter]
        else:
            single = tok.encode(letter, add_special_tokens=False)
            assert len(single) == 1 and tok.decode(single) == letter
            complete = prefix['input_ids'] + single + tok.encode(')', add_special_tokens=False)
        assert complete[:n] == prefix['input_ids']
        assert tok.decode([complete[n]]).strip() == letter
        assert tok.decode(complete).rstrip().endswith('('+letter+')')
        answer_ids[letter] = complete[n]
        completions[letter] = complete
    assert answer_ids['A'] != answer_ids['B']
    return {**prefix, 'answer_ids':answer_ids, 'extraction_position':len(prefix['input_ids']),
            'completion_ids':completions, 'whole_string_reencoding_prefix_stable':reencoded_stable}


def views(group):
    return symmetrize_questions(pd.DataFrame([group['views'][0]])).to_dict('records')
