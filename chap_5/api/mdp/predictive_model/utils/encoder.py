BITS_PER_ITEM = 15
MASK_15_BITS = (1 << BITS_PER_ITEM) - 1


def as_tuple(s) -> tuple:
    s = (s,) if isinstance(s, int) else s
    return tuple(int(x) for x in s)


def encode(items) -> bytes:
    """Encode un tuple d'entiers en bytes (BYTEA) compact."""
    items = as_tuple(items)
    k = len(items)
    packed_int = 0
    for item in items:
        packed_int = (packed_int << BITS_PER_ITEM) | item

    # Calcul du nombre d'octets nécessaires (ex: 45 bits -> 6 octets)
    num_bits = k * BITS_PER_ITEM
    num_bytes = (num_bits + 7) // 8
    return packed_int.to_bytes(num_bytes, 'big')


def decode(s_bytes: bytes) -> tuple:
    """Décode des bytes (BYTEA) en tuple d'entiers."""
    if not s_bytes:
        return ()

    packed_int = int.from_bytes(s_bytes, 'big')

    # On déduit k à partir de la taille en octets
    num_bytes = len(s_bytes)
    num_bits = num_bytes * 8
    k = num_bits // BITS_PER_ITEM

    items = []
    for _ in range(k):
        items.append(packed_int & MASK_15_BITS)
        packed_int >>= BITS_PER_ITEM

    return tuple(reversed(items))