"""Native executor adapter of official fixed-R greedy WaveFront control flow.
Independent scheduler, not alias to continue_pipelined. Fixed draft depth dividing R; greedy.
"""
import json,os
from pathlib import Path
from .pipeline import PipelineLane

def continue_wavefront(first_token,maximum,executor):
 s=executor.schedule
 assert s.second_step is None and 0<s.step_size<s.stages and s.stages%s.step_size==0
 committed=[first_token];active=[PipelineLane(stage=0,index=1,token=first_token)];next_branch=1;tick=0;events=[]
 counters=dict(wfd_proposed=0,wfd_accepted=0,wfd_rejected=0,wfd_boundary=0,wfd_ticks=0)
 trace=os.environ.get('WFD_TRACE_DIR')
 def emit(kind,**detail):
  if trace:events.append(dict(tick=tick,event=kind,committed=list(committed),active=[dict(position=x.index,token=x.token,depth=x.stage,branch=x.branch,parent=x.parent_branch) for x in active],**detail))
 def child(parent,token,origin):
  nonlocal next_branch
  x=PipelineLane(stage=0,index=parent.index+1,token=token,branch=next_branch,parent_branch=parent.branch,origin_stage=origin);next_branch+=1;return x
 if maximum==1:executor.retain_branches(());return committed
 while len(committed)<maximum:
  tick+=1;active.sort(key=lambda x:x.index);before=list(active);results=list(executor.forward(before));assert len(before)==len(results)
  verifies=[];drafts=[];following=[]
  for x,r in zip(before,results):
   x.stage+=s.step_size;x.state=r.state
   if x.stage==s.stages:verifies.append((x,r))
   else:
    following.append(x)
    if x.stage==s.step_size:
     assert r.proposed;x.first_token=r.prediction;x.first_state=r.proposal_state;drafts.append((x,r))
  assert len(verifies)<=1 and len(drafts)<=1
  active=following;rejected=False;accepted_stage=None
  emit('advance',verify_count=len(verifies),draft_count=len(drafts))
  for x,r in verifies:
   assert x.index==len(committed)
   nxt=next((t for t in active if t.index==x.index+1),None)
   accepted=nxt is not None and nxt.token==r.prediction
   emit('verify',gold=r.prediction,candidate=None if nxt is None else nxt.token,accepted=accepted)
   committed.append(r.prediction)
   if nxt is None:
    counters['wfd_boundary']+=1
    if len(committed)<maximum:active.append(child(x,r.prediction,s.stages-1))
   elif accepted:
    counters['wfd_proposed']+=1;counters['wfd_accepted']+=1;accepted_stage=s.proposal_stage
   else:
    counters['wfd_proposed']+=1;counters['wfd_rejected']+=1;rejected=True
    active=[t for t in active if t.index<=x.index]
    if len(committed)<maximum:active.append(child(x,r.prediction,s.stages-1))
   emit('commit',accepted=accepted,rejected=rejected,boundary=nxt is None)
   if executor.should_stop(committed) or len(committed)>=maximum:
    active=[];executor.retain_branches(());break
  if not active:break
  # Official scheduler skips draft births on a rejecting tick.
  if not rejected:
   for x,r in drafts:
    assert not any(t.index==x.index+1 for t in active)
    active.append(child(x,r.prediction,s.proposal_stage));emit('proposal',prediction=r.prediction)
  executor.retain_branches(tuple(active),accepted_stage)
  assert len({t.index for t in active})==len(active)
  assert len(active)<=s.max_rows
 counters['wfd_ticks']=tick;executor.metrics.update(counters)
 if trace:
  p=Path(trace);p.mkdir(parents=True,exist_ok=True);(p/f'trace_{os.getpid()}_{len(list(p.glob("trace_*.json"))):04d}.json').write_text(json.dumps(dict(output=committed,events=events,counters=counters),indent=2))
 return committed
