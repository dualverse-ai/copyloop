import os

BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80)


def bucket_width(real):
    if os.environ.get('LOOPSPEC_BUCKET_GRAPHS') != '1':
        return real
    if not 1 <= real <= 80:
        raise ValueError(f'Invalid bucket width {real}; W_CAP=80')
    return next(width for width in BUCKETS if width >= real)


def admitted_width(live, proposed, cap=80):
    if live + proposed <= cap:
        return proposed, False
    if live + 1 > cap:
        raise RuntimeError('No room for mandatory draft row')
    return 1, True
