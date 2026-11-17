"""Independent CAA-scope Linear hook and single-letter evaluator.

No Remy imports, normalization, clipping, symmetrization, or sequence scoring.
"""
import torch


def chat_tokens(tokenizer, question, answer='('):
    return tokenizer.encode(f'[INST] {question.strip()} [/INST] {answer.strip()}')


def instruction_boundary(tokens, tokenizer):
    # Match official encode(E_INST)[1:] exactly, including its BOS convention.
    marker = tokenizer.encode('[/INST]')[1:]
    if not marker:
        raise ValueError('Empty instruction marker')
    for start in range(len(tokens) - len(marker), -1, -1):
        if tokens[start:start + len(marker)] == marker:
            return start + len(marker) - 1
    raise ValueError('Instruction delimiter missing')


def translation_hook(vector, alpha, boundary):
    def hook(module, args, kwargs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        positions = kwargs.get('position_ids')
        if positions is None:
            raise ValueError('position_ids required; cannot infer cached positions')
        mask = (positions >= boundary).unsqueeze(-1).float()
        # Official helper uses float32 mask multiplication and in-place add.
        hidden += mask * (float(alpha) * vector.to(hidden.device))
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
    return hook


def score_logits(logits, a_id, b_id, matching):
    if matching not in ('(A)', '(B)') or a_id == b_id:
        raise ValueError('Invalid A/B labels')
    values = logits[0, -1].float()
    if not torch.isfinite(values).all():
        raise FloatingPointError('Nonfinite logits')
    probs = values.softmax(-1)
    # Stable equivalent of p(target)/(p(A)+p(B)); never subtract sequence sums.
    margin = values[a_id].double() - values[b_id].double()
    if matching == '(B)':
        margin = -margin
    probability = margin.sigmoid().item()
    return {'normalized_target_probability': probability,
            'behavior_matching_answer': float(margin > 0),
            'target_logit_margin': margin.item(),
            'p_a_full_vocab': probs[a_id].item(), 'p_b_full_vocab': probs[b_id].item()}
