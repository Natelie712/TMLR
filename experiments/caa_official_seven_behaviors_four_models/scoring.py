"""Scheme B: softmax over ALL printed option-label tokens at one decision position.

Scheme A is retained as a separate, explicitly named diagnostic. No sequence
likelihoods, implicit relabeling, or selected contrast-pair denominators.
"""
import re
import torch


def parse_options(question):
    heading=re.search(r'^Choices:[ \t]*$',question,re.M)
    text=question[heading.end():] if heading else question
    markers=list(re.finditer(r'^[ \t]*\(([A-Z])\)[ \t]*',text,re.M))
    labels=[m[1] for m in markers]
    if len(labels)<2 or len(set(labels))!=len(labels):
        raise ValueError('Expected at least two distinct printed option labels')
    result={m[1]:text[m.end():markers[i+1].start() if i+1<len(markers) else None].strip() for i,m in enumerate(markers)}
    if any(not v for v in result.values()):raise ValueError('Empty option text')
    return result


def target_label(value):
    m=re.fullmatch(r'\(([A-Z])\)',value)
    if not m:raise ValueError('Expected a source label such as (C)')
    return m[1]


def score(logits, option_ids, target):
    """FP32 raw logits/full-vocabulary softmax; FP64 selected-logit softmax."""
    x=logits.reshape(-1,logits.shape[-1])[-1].float()
    if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite logits')
    if target not in option_ids:raise ValueError('Target is absent from printed options')
    labels=list(option_ids);ids=list(option_ids.values())
    if len(ids)<2 or len(set(ids))!=len(ids):raise ValueError('Candidate token IDs must be distinct')
    if min(ids)<0 or max(ids)>=len(x):raise ValueError('Candidate token outside vocabulary')
    selected=x[ids].double().softmax(0)
    full=x.softmax(0)
    probs={k:float(selected[i]) for i,k in enumerate(labels)}
    result=dict(target=target,option_labels=labels,candidate_token_ids=option_ids,
        candidate_logits={k:float(x[v]) for k,v in option_ids.items()},
        full_vocab_option_probabilities={k:float(full[v]) for k,v in option_ids.items()},
        normalized_option_probabilities=probs,normalized_target_probability=probs[target],
        full_vocab_target_probability=float(full[option_ids[target]]),
        all_option_probability_mass=float(full[ids].double().sum()),
        logsumexp_full_vocab=float(x.double().logsumexp(0)))
    if 'A' in option_ids and 'B' in option_ids:
        a,b=option_ids['A'],option_ids['B'];den=float(full[a])+float(full[b])
        # Exact saved-reference aggregation, including the unsupported-target zero.
        orig=(float(full[option_ids[target]])/den if den>0 else None) if target in ['A','B'] else 0.
        stable=float((x[a].double()-x[b].double()).sigmoid()) if target=='A' else float((x[b].double()-x[a].double()).sigmoid()) if target=='B' else 0.
        result.update(scheme_A_original_probability=orig,scheme_A_stable_probability=stable,
                      ab_probability_mass=den,scheme_A_target_supported=target in ['A','B'],
                      B_minus_A_original=None if orig is None else probs[target]-orig,
                      B_minus_A_stable=probs[target]-stable)
    return result
