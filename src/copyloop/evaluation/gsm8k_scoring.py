"""lm-eval 0.4.12 GSM8K regex and exact-match semantics."""
import re

def extract(response, strict=False):
 pattern = r'The answer is (\-?[0-9\.\,]+).' if strict else r'(-?[$0-9.,]{2,})|(-?[0-9]+)'
 matches=re.findall(pattern,response)
 if not matches:return '[invalid]'
 match=matches[0 if strict else -1]
 if isinstance(match,tuple):match=next((m for m in match if m),'[invalid]')
 return match.strip()

def correct(prediction,target):
 def normalize(s):
  for regex in [',',r'\$',r'(?s).*#### ',r'\.$']:s=re.sub(regex,'',s)
  return s.lower()
 return normalize(prediction)==normalize(target)
