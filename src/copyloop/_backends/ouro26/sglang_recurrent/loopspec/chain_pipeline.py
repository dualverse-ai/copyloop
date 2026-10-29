"""Grouped greedy lookup chains over native per-position KV ownership."""
from dataclasses import dataclass
import os
import time

from .pipeline import PipelineLane
from .lookup_chain import draft_lookup_chain


@dataclass
class Chain:
    identity: int
    parent: int | None
    origin: int
    prefix: list[int]
    lanes: list[PipelineLane]


def continue_chains(first_token, maximum, executor):
    schedule = executor.schedule
    committed = [first_token]
    next_branch = 0
    next_group = 0
    parents = {}
    hist = {}
    emitted = []
    draft_seconds = 0.0
    truncation_events = []
    renewal = []
    groups = []
    overflow_events = 0
    serialized_frontiers = 0
    flatten = os.environ.get("LOOPSPEC_CHAIN_FLATTEN", "0") == "1"
    truncate = os.environ.get("LOOPSPEC_CHAIN_TRUNCATE", "") == "shallow"

    def create(prefix, token, parent_branch, parent_group, origin, live_rows=0):
        nonlocal next_branch, next_group, draft_seconds, overflow_events
        event_k = schedule.chain_k
        start = time.perf_counter()
        chain_tokens = [token] + draft_lookup_chain(
            executor.request.prompt_ids + prefix + [token], event_k)
        draft_seconds += time.perf_counter() - start
        chain_tokens = chain_tokens[:maximum - len(prefix)]
        if schedule.second_step is None:
            from .buckets import admitted_width
            width, overflow = admitted_width(live_rows, len(chain_tokens))
            chain_tokens = chain_tokens[:width]
            overflow_events += int(overflow)
        elif truncate:
            cap = int(os.environ.get("LOOPSPEC_CHAIN_W_CAP", "80"))
            if live_rows + len(chain_tokens) > cap:
                chain_tokens = chain_tokens[:1]
                overflow_events += 1
        lanes = []
        for offset, value in enumerate(chain_tokens):
            lane = PipelineLane(stage=0, index=len(prefix) + offset + 1,
                                token=value, branch=next_branch,
                                parent_branch=parent_branch, origin_stage=origin)
            lanes.append(lane)
            parent_branch = next_branch
            next_branch += 1
        group = Chain(next_group, parent_group, origin, list(prefix), lanes)
        parents[next_group] = parent_group
        next_group += 1
        return group

    def descends(group, root):
        identity = group.identity
        while identity is not None:
            if identity == root:
                return True
            identity = parents.get(identity)
        return False

    groups = [create([], first_token, None, None, -1)]
    while len(committed) < maximum:
        following = []
        children = []
        terminal = None
        if flatten:
            offsets = []
            live_lanes = []
            for group in groups:
                offsets.append((group, len(live_lanes), len(group.lanes)))
                live_lanes.extend(group.lanes)
            cap = int(os.environ.get("LOOPSPEC_CHAIN_W_CAP", "80"))
            if len(live_lanes) > schedule.max_rows:
                raise RuntimeError(f"flattened frontier exceeds scheduler capacity: {len(live_lanes)} > {schedule.max_rows}")
            if len(live_lanes) > cap:
                serialized_frontiers += 1
                work = [(group, executor.forward(group.lanes)) for group in groups]
                executor.metrics["chain_max_frontier_width"] = max(
                    executor.metrics.get("chain_max_frontier_width", 0),
                    max((len(group.lanes) for group in groups), default=0),
                )
            else:
                flat_results = executor.forward(live_lanes)
                executor.metrics["chain_max_frontier_width"] = max(
                    executor.metrics.get("chain_max_frontier_width", 0), len(live_lanes)
                )
                work = [(group, flat_results[start:start + count]) for group, start, count in offsets]
        else:
            work = [(group, executor.forward(group.lanes)) for group in groups]
        for group, results in work:
            end_depth = group.lanes[0].stage + schedule.step_size
            for lane, result in zip(group.lanes, results, strict=True):
                lane.state = result.state
                lane.stage = end_depth
            if end_depth == schedule.stages:
                if terminal is not None:
                    raise RuntimeError('multiple final chain roots')
                terminal = group, results
                continue
            proposal_index = len(group.lanes) - 1
            if truncate and end_depth - 1 == schedule.proposal_stage and results[-1].proposed:
                proposal_index = next(
                    (i for i in range(len(group.lanes) - 1)
                     if results[i].prediction != group.lanes[i + 1].token),
                    len(group.lanes) - 1,
                )
                old_width = len(group.lanes)
                group.lanes = group.lanes[:proposal_index + 1]
                truncation_events.append({"m": proposal_index, "old_width": old_width,
                                          "new_width": len(group.lanes)})
            last = group.lanes[proposal_index]
            prediction = results[proposal_index]
            if end_depth - 1 in (schedule.proposal_stage, schedule.second_stage) and prediction.proposed:
                if end_depth - 1 == schedule.proposal_stage:
                    last.first_token = prediction.prediction
                else:
                    last.second_token = prediction.prediction
                prefix = group.prefix + [lane.token for lane in group.lanes[:proposal_index + 1]]
                if len(prefix) < maximum:
                    children.append(create(prefix, prediction.prediction, last.branch,
                                           group.identity, end_depth - 1,
                                           sum(len(active.lanes) for active in groups if active.lanes[0].stage < schedule.stages)
                                           + sum(len(child.lanes) for child in children)))
            following.append(group)
        following.extend(children)
        if terminal is not None:
            group, results = terminal
            if group.lanes[0].index != len(committed):
                raise RuntimeError('chain verification out of order')
            accepted = 0
            consumed = 0
            mismatch = False
            for offset, result in enumerate(results):
                committed.append(result.prediction)
                consumed += 1
                if offset + 1 < len(group.lanes):
                    mismatch = result.prediction != group.lanes[offset + 1].token
                    if not mismatch:
                        accepted += 1
                stop = executor.should_stop(committed) or len(committed) == maximum
                if stop or mismatch:
                    break
            hist[str(accepted)] = hist.get(str(accepted), 0) + 1
            emitted.append(consumed)
            if stop:
                executor.retain_branches(())
                break
            selected = None
            if not mismatch:
                selected = next((child for child in following
                                 if child.parent == group.identity
                                 and child.lanes[0].token == committed[-1]), None)
            renewal.append({'n': consumed, 'continue': selected is not None,
                            'cause': 'continue' if selected is not None else
                            ('shallow_deep_disagreement' if mismatch or len(group.lanes) > 1 else 'self_draft_wrong'),
                            'm': len(group.lanes) - 1})
            if selected is None:
                selected = create(committed[:-1], committed[-1],
                                  group.lanes[consumed - 1].branch, group.identity,
                                  schedule.stages - 1,
                                  (sum(len(active.lanes) for active in following)
                                   if schedule.second_step is not None else 0))
                following.append(selected)
            groups = [child for child in following if descends(child, selected.identity)]
        else:
            groups = following
        lanes = [lane for group in groups for lane in group.lanes]
        if len(lanes) > schedule.max_rows:
            raise RuntimeError('chain state capacity exceeded')
        executor.retain_branches(lanes)
    executor.metrics.update(chain_acceptance_histogram=hist,
                            chain_verification_events=len(emitted),
                            chain_committed_tokens=sum(emitted),
                            chain_events_m_positive=sum(event['m'] > 0 for event in renewal),
                            chain_sum_m=sum(event['m'] for event in renewal),
                            chain_continues=sum(event['continue'] for event in renewal),
                            chain_restarts=sum(not event['continue'] for event in renewal),
                            chain_overflow_events=overflow_events,
                            chain_serialized_frontier_steps=serialized_frontiers,
                            drafter_seconds=draft_seconds,
                            chain_truncation_events=len(truncation_events))
    return committed
