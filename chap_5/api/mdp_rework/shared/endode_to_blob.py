from chap_5.api.mdp_rework.config import BITS_MOVIE_REPRESENTATION

MASK_15_BITS = (1 << BITS_MOVIE_REPRESENTATION) - 1

def as_tuple(s):
    return (s,) if isinstance(s, int) else tuple(map(int, s))

def encode(items):
    items = as_tuple(items)
    packed_int = 0
    for item in items:
        packed_int = (packed_int << BITS_MOVIE_REPRESENTATION) | item
    return packed_int.to_bytes((len(items) * BITS_MOVIE_REPRESENTATION + 7) // 8, 'big')

def decode(s_bytes):
    if not s_bytes:
        return ()
    packed_int = int.from_bytes(s_bytes, 'big')
    k = len(s_bytes) * 8 // BITS_MOVIE_REPRESENTATION
    return tuple((packed_int >> (i * BITS_MOVIE_REPRESENTATION)) & MASK_15_BITS for i in reversed(range(k)))