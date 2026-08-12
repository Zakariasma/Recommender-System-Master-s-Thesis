import struct
import numpy as np

_state_struct_cache: dict[int, struct.Struct] = {}
_proba_struct_cache: dict[int, struct.Struct] = {}

def _get_state_struct(n: int) -> struct.Struct:
    s = _state_struct_cache.get(n)
    if s is None:
        s = _state_struct_cache[n] = struct.Struct(f'>{n}H')
    return s

def _get_proba_struct(n: int) -> struct.Struct:
    s = _proba_struct_cache.get(n)
    if s is None:
        s = _proba_struct_cache[n] = struct.Struct(f'>{n}f')
    return s

def encode_state(s_tuple: tuple) -> bytes:
    if not s_tuple:
        return b''
    return _get_state_struct(len(s_tuple)).pack(*s_tuple)

def decode_state(blob: bytes) -> tuple:
    if not blob:
        return ()
    return _get_state_struct(len(blob) // 2).unpack(blob)

def encode_proba_list(probas: list) -> bytes:
    if not probas:
        return b''
    return _get_proba_struct(len(probas)).pack(*probas)

def decode_proba_list(blob: bytes) -> np.ndarray:
    if not blob:
        return np.array([], dtype='>f4')
    return np.frombuffer(blob, dtype='>f4')

def encode_successors_fixed(s_primes: list) -> bytes:
    if not s_primes:
        return b''
    flat = [v for s in s_primes for v in s]
    return _get_state_struct(len(flat)).pack(*flat)

def decode_successors_fixed(blob: bytes, k: int) -> np.ndarray:
    if not blob or len(blob) < k * 2:
        return np.empty((0, k), dtype='>u2')
    return np.frombuffer(blob, dtype='>u2').reshape(-1, k)