# Verbatim from dreams/utils/misc.py (pluskal-lab/DreaMS @ 02be141); only the
# helper that feed_forward.py references is kept.


def interpolate_interval(a, b, n, only_inter=False, rounded=False):
    """
    :param a: Start point.
    :param b: End point.
    :param n: Num. of steps.
    :param only_inter: Does not return interval ends a and b.
    :param rounded:
    """
    x_min, x_max = min(a, b), max(a, b)
    res = [x_min + i * (x_max - x_min) / (n + 1) for i in range(1 if only_inter else 0, n + 1 if only_inter else n + 2)]
    if x_max == a:
        res.reverse()
    if rounded:
        res = [round(x) for x in res]
    return res
