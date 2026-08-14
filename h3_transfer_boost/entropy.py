import math
from collections import Counter


def entropy_bits_per_byte(data):
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def estimated_ratio(entropy):
    return min(1.0, max(0.0, entropy / 8.0))
