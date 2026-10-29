"""Greedy longest-suffix lookup proposals and full-depth chain verification."""


def next_lookup_token(context, minimum_suffix=1, maximum_suffix=4):
    """Return the continuation after the most recent matching suffix, or None."""
    values = list(context)
    for width in range(min(maximum_suffix, len(values)), minimum_suffix - 1, -1):
        suffix = values[-width:]
        for start in range(len(values) - width - 1, -1, -1):
            if values[start : start + width] == suffix:
                continuation = start + width
                if continuation < len(values):
                    return values[continuation]
                break
    return None


def draft_lookup_chain(context, limit=8):
    """D1: longest matching suffix, most recent occurrence, static continuation."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    values = list(context)
    # Bijective token-ID to Unicode encoding, C-level exact suffix search.
    text = ''.join(map(chr, values))
    for width in range(min(4, len(values)), 0, -1):
        start = text.rfind(text[-width:], 0, len(text) - 1)
        if start >= 0:
            return values[start + width:start + width + limit]
    return []



def verify_lookup_chain(proposed, target_predictions):
    """Return matched proposal count, emitted tokens, and first mismatch index.

    target_predictions contains the full-depth argmax conditioned on each token
    in proposed followed by one continuation prediction, when available.
    """
    accepted = 0
    for proposed_token, target_token in zip(proposed, target_predictions):
        if proposed_token != target_token:
            return accepted, list(proposed[:accepted]) + [target_token], accepted
        accepted += 1
    if len(target_predictions) > len(proposed):
        return accepted, list(proposed) + [target_predictions[len(proposed)]], None
    return accepted, list(proposed[:accepted]), None
