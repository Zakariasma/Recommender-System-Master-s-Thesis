def as_tuple(s) -> tuple:
    s = (s,) if isinstance(s, int) else s
    return tuple(int(x) for x in s)


def encode(items) -> str:
    return ','.join(str(x) for x in items)


def decode(s_str: str) -> tuple:
    return tuple(int(x) for x in s_str.split(',') if x)