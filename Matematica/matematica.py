from itertools import permutations

# ----- Helper functions -----
def compose(p, q):
    """Return composition p ∘ q (apply q first, then p)."""
    return tuple(p[i - 1] for i in q)

def inverse(p):
    """Return the inverse of permutation p."""
    inv = [0] * len(p)
    for i, v in enumerate(p, start=1):
        inv[v - 1] = i
    return tuple(inv)

def parity(p):
    """Return +1 for even, -1 for odd permutation."""
    inversions = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                inversions += 1
    return 1 if inversions % 2 == 0 else -1

def to_cycles(p):
    """Convert permutation tuple to cycle notation string."""
    n = len(p)
    seen = [False] * n
    cycles = []
    for i in range(n):
        if not seen[i]:
            cycle = []
            j = i
            while not seen[j]:
                seen[j] = True
                cycle.append(j + 1)
                j = p[j] - 1
            if len(cycle) > 1:
                cycles.append(tuple(cycle))
    if not cycles:
        return "()"
    return " ".join("(" + " ".join(map(str, c)) + ")" for c in cycles)

# ----- Generate A5 -----
elements = [1, 2, 3, 4, 5]
A5 = [p for p in permutations(elements) if parity(p) == 1]

# ----- Conjugacy class finder -----
def conjugacy_class(g, group):
    """Return the conjugacy class of g within given group."""
    return {compose(compose(h, g), inverse(h)) for h in group}

# ----- Identify all 5-cycles -----
def is_5_cycle(p):
    seen = [False] * len(p)
    count = 0
    for i in range(len(p)):
        if not seen[i]:
            count += 1
            j = i
            length = 0
            while not seen[j]:
                seen[j] = True
                j = p[j] - 1
                length += 1
            if length != len(p) and length != 1:
                return False
    return count == 1  # single 5-cycle

five_cycles = [p for p in A5 if is_5_cycle(p)]

# ----- Find distinct conjugacy classes of 5-cycles -----
classes = []
remaining = set(five_cycles)

while remaining:
    g = remaining.pop()
    cls = conjugacy_class(g, A5)
    cls &= set(five_cycles)  # keep only 5-cycles
    classes.append(cls)
    remaining -= cls

# ----- Display results -----
for i, c in enumerate(classes, start=1):
    print(f"\n=== 5-cycle Class {i} (size {len(c)}) ===")
    for elem in sorted(c):
        print(to_cycles(elem))