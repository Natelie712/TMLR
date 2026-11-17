"""Native model adapters for all-option scoring; original CAA code is immutable."""
import json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/caa_transfer_full_20261027'))
import transfer_common as tc
from scoring import parse_options,target_label,score
from audit_test_datasets import CONCEPTS
MODELS=json.loads((ROOT/'experiments/caa_transfer_full_20261027/experiment_config.json').read_text())['models']
VARIANTS=['legacy_parenthesis','assistant_letter','assistant_answer_parenthesis','system_letter']
INSTRUCTION='Answer the multiple-choice question with exactly one uppercase option letter. Do not add any explanation.'


def data(concept,split):
    name='generate_dataset.json' if split=='generate' else 'test_dataset_ab.json'
    path=HERE/'upstream/datasets'/split/concept/name
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    assert tc.sha(path)==manifest['files'][f'datasets/{split}/{concept}/{name}']['sha256']
    return json.loads(path.read_text())


def decision(tok,key,question,variant):
    suffix={'legacy_parenthesis':'(','assistant_letter':'','assistant_answer_parenthesis':'The answer is (','system_letter':''}[variant]
    if variant=='system_letter':
        assert key!='llama2'
        rendered=tok.apply_chat_template([dict(role='system',content=INSTRUCTION),dict(role='user',content=question.strip())],
            tokenize=False,add_generation_prompt=True,enable_thinking=False,date_string='07 Jan 2026')
        ids=tok.encode(rendered,add_special_tokens=False)
        marker={'llama3b':'<|eot_id|>','phi':'<|end|>','qwen3':'<|im_end|>'}[key]
        boundary=max(i for i,v in enumerate(ids) if v==tok.convert_tokens_to_ids(marker))
        p=dict(input_ids=ids,rendered=rendered,boundary=boundary)
    else:p=tc.legacy.prompt(tok,key,question,answer=suffix)
    options=parse_options(question);aids={};completions={};diagnostics={}
    for letter in options:
        bare=tok.convert_tokens_to_ids(letter)
        assert bare!=tok.unk_token_id and tok.decode([bare])==letter,(key,letter,bare)
        closing=[tok.convert_tokens_to_ids(')')] if suffix.endswith('(') else []
        complete=p['input_ids']+[bare]+closing
        if key in ['llama2','phi'] and variant=='legacy_parenthesis':
            expected=tc.legacy.prompt(tok,key,question,'('+letter+')')['input_ids']
            assert expected[:len(p['input_ids'])]==p['input_ids'] and expected[len(p['input_ids'])]==bare,(key,letter)
            complete=expected
        assert tok.decode(complete).rstrip().endswith(letter+')' if closing else letter)
        aids[letter]=bare;completions[letter]=complete
        spaced=tok.encode(' '+letter,add_special_tokens=False)
        reencoded=tok.encode(p['rendered']+letter+(')' if closing else ''),add_special_tokens=(key=='llama2'))
        diagnostics[letter]=dict(bare_id=bare,bare_decoded=tok.decode([bare]),isolated_ids=tok.encode(letter,add_special_tokens=False),
            spaced_ids=spaced,spaced_decoded=tok.decode(spaced),
            full_string_prefix_stable=reencoded==complete)
    return dict(**p,option_ids=aids,completion_ids=completions,extraction_position=len(p['input_ids']),
                tokenization=diagnostics,prompt_variant=variant,question_preserved=True)


def task(tok,key,concept,index,row,variant,split):
    d=decision(tok,key,row['question'],variant)
    t=target_label(row['answer_matching_behavior']);n=target_label(row['answer_not_matching_behavior'])
    assert t in d['option_ids'] and n in d['option_ids'] and t!=n
    return dict(concept=concept,index=index,split=split,target=t,nonmatching=n,question=row['question'],
                decision=d,ids=d['input_ids'])
