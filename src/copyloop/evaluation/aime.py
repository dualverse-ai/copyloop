"""Strict final-only AIME integer extraction, not an LLM-judge reproduction."""
import re

def extract_integer(text, *, bounded=True):
    # Last boxed answer follows standard competition-math final-answer convention.
    boxes = re.findall(r'\\boxed\s*\{([^{}]*)\}', text)
    if boxes:
        candidate = boxes[-1].strip()
        origin = 'last_boxed'
    else:
        explicit = re.findall(r'(?:final\s+answer|answer)\s*(?:is|:|=)\s*\$?\s*([+-]?\d+(?:\.\d+)?)\s*\$?(?=[\s.,;!]|$)', text, re.I)
        if explicit:
            candidate = explicit[-1];origin='explicit_answer'
        else:
            candidate = text.strip().strip('$').strip();origin='bare_final'
    if not re.fullmatch(r'\d{1,3}' if bounded else r'[+-]?\d+',candidate):
        return {'value':None,'parse_ok':False,'source':origin,'candidate':candidate}
    value=int(candidate)
    return {'value':value,'parse_ok':(0<=value<=999) if bounded else True,'source':origin,'candidate':candidate}

def score_integer(text,target, *, bounded=True):
    pred=extract_integer(text,bounded=bounded)
    gold=str(target).strip()
    if not re.fullmatch(r'\d{1,3}' if bounded else r'[+-]?\d+',gold):raise ValueError(f'invalid AIME target {target!r}')
    return {'correct':bool(pred['parse_ok'] and pred['value']==int(gold)),
            'prediction_parse_ok':pred['parse_ok'],'prediction':pred,'target_integer':int(gold),
            'scorer':('strict final integer 0..999' if bounded else 'strict final signed integer')+'; last boxed else explicit answer else bare'}
