import struct

from chap_5.api.mdp_rework.config import BITS_MOVIE_REPRESENTATION

MASK_BITS = (1 << BITS_MOVIE_REPRESENTATION) - 1

def as_tuple(s):
    return (s,) if isinstance(s, int) else tuple(map(int, s))

def encode(items):
    items = as_tuple(items)
    packed_int = 0
    for item in items:
        packed_int = (packed_int << BITS_MOVIE_REPRESENTATION) | item
    total_bits = len(items) * BITS_MOVIE_REPRESENTATION
    return packed_int.to_bytes((total_bits + 7) // 8, 'big')

def decode(s_bytes):
    if not s_bytes:
        return ()
    packed_int = int.from_bytes(s_bytes, 'big')
    k = len(s_bytes) * 8 // BITS_MOVIE_REPRESENTATION
    return tuple((packed_int >> (i * BITS_MOVIE_REPRESENTATION)) & MASK_BITS for i in reversed(range(k)))


def unpack_transitions(succ_blob: bytes, proba_blob: bytes) -> list:
    """
    Décode les BLOBs structurés générés par create_row_from_transitions.
    Retourne une liste de tuples (s_prime_bytes, proba).
    """
    transitions = []

    offset_succ = 0
    offset_proba = 0

    if len(succ_blob) < 4:
        return transitions

    # Lire le nombre de tailles k différentes
    num_k = struct.unpack_from('i', succ_blob, offset_succ)[0]
    offset_succ += 4
    offset_proba += 4

    for _ in range(num_k):
        # Lire la valeur de k (pour info, mais on en a surtout besoin pour calculer la taille de s_prime)
        k = struct.unpack_from('i', succ_blob, offset_succ)[0]
        offset_succ += 4
        offset_proba += 4

        # Lire le nombre de transitions pour ce k
        num_trans = struct.unpack_from('i', succ_blob, offset_succ)[0]
        offset_succ += 4
        offset_proba += 4

        # Taille en octets d'un état s_ pour ce k
        state_byte_size = (k * BITS_MOVIE_REPRESENTATION + 7) // 8

        for _ in range(num_trans):
            s_prime = succ_blob[offset_succ: offset_succ + state_byte_size]
            proba = struct.unpack_from('f', proba_blob, offset_proba)[0]

            transitions.append((s_prime, proba))

            offset_succ += state_byte_size
            offset_proba += 4

    return transitions