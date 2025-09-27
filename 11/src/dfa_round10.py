# -*- coding: utf-8 -*-
"""Differential Fault Attack on AES-128 with a single-byte fault at MixColumns input in Round 9.
We recover the last round key bytes (Round 10), then invert the key schedule to get the MAIN key.
Logic follows the hints: group bytes per column after InvShiftRows; for each pair, test 4 faulty-row
hypotheses and aggregate consistent last-round key candidates via the relations:
D = InvSBox(C^k) ^ InvSBox(C'^k)  (computed per byte)
and for the correct hypothesis there exists Δ!=0 such that D matches one of the 4 MixColumns patterns:
p=0: [2Δ,1Δ,1Δ,3Δ], p=1: [3Δ,2Δ,1Δ,1Δ], p=2: [1Δ,3Δ,2Δ,1Δ], p=3: [1Δ,1Δ,3Δ,2Δ].
"""
import numpy as np
from .aes_tables import INV_SBOX, mul2, mul3

INV_SBOX_A = np.array(INV_SBOX, dtype=np.uint8)
KS = np.arange(256, dtype=np.uint8)  # 0..255 for vectorized key guesses

# Ciphertext indices for each column *before* ShiftRows (i.e., after InvShiftRows).
# Given ciphertext is after ShiftRows, so bytes of one MC column are at indices:
# idx[r] = r + 4*((col + r) % 4) for r in 0..3
COL_IDXS = [[r + 4*((c + r) % 4) for r in range(4)] for c in range(4)]

# For each hypothetical faulty row p, which rows have coeff 1, which coeff 2 or 3?
# patterns arrays provide the coefficient vector [c0,c1,c2,c3] for rows 0..3.
PATTERNS = {
    0: [2,1,1,3],
    1: [3,2,1,1],
    2: [1,3,2,1],
    3: [1,1,3,2],
}

def _mulc(coeff, delta):
    if coeff == 1: return delta
    if coeff == 2: return mul2(delta)
    if coeff == 3: return mul3(delta)
    raise ValueError("coeff must be 1..3")

def accumulate_counts_for_pair(C16, F16, counts_16x256):
    """Update vote counts for last-round key bytes using a single (C,F) pair.
    counts_16x256: array int32 shape (16,256) accumulated across pairs.
    """
    # For each column
    for col in range(4):
        idxs = COL_IDXS[col]  # 4 ciphertext indices belonging to this column
        c_bytes = [int(C16[i]) for i in idxs]
        f_bytes = [int(F16[i]) for i in idxs]

        # Precompute D_k for 4 positions: arrays (256,) each
        Dks = []
        for b in range(4):
            c = c_bytes[b]; f = f_bytes[b]
            d = INV_SBOX_A[np.bitwise_xor(c, KS)] ^ INV_SBOX_A[np.bitwise_xor(f, KS)]
            Dks.append(d)  # (256,)

        # Test 4 hypotheses for faulty row p
        for p in range(4):
            coeffs = PATTERNS[p]
            ones_pos = [i for i,c in enumerate(coeffs) if c == 1]
            c2_pos   = coeffs.index(2)
            c3_pos   = coeffs.index(3)

            # Common Δ candidates are values appearing in both ones positions (and ≠ 0)
            vals1 = Dks[ones_pos[0]]
            vals2 = Dks[ones_pos[1]]
            # Build hist of values for each (value -> mask of keys)
            # We'll iterate unique Δ in intersection to reduce cost
            uniq1 = np.unique(vals1)
            uniq2 = np.unique(vals2)
            common = np.intersect1d(uniq1, uniq2, assume_unique=False)
            common = common[common != 0]  # Δ must be nonzero

            if common.size == 0:
                continue

            # For each Δ in common, compute masks and update counts for consistent keys
            for delta in common.tolist():
                # masks for keys at each position
                m1 = (vals1 == delta)  # keys for ones_pos[0]
                m2 = (vals2 == delta)  # keys for ones_pos[1]
                # For coeff2 and coeff3 targets
                t2 = _mulc(2, int(delta))
                t3 = _mulc(3, int(delta))
                m_c2 = (Dks[c2_pos] == t2)
                m_c3 = (Dks[c3_pos] == t3)

                # If any mask is empty, skip this Δ quickly
                if not (m1.any() and m2.any() and m_c2.any() and m_c3.any()):
                    continue

                # Weight to avoid overcounting highly ambiguous deltas
                # weight = 1 / (|S1| + |S2| + |S2coeff| + |S3coeff|)
                denom = int(m1.sum()) + int(m2.sum()) + int(m_c2.sum()) + int(m_c3.sum())
                if denom <= 0:
                    continue
                w = 1.0 / denom

                # Update counts for keys that satisfy at each byte position (add fractional votes)
                counts_16x256[idxs[ones_pos[0]], :] += (m1 * w).astype(np.float32)
                counts_16x256[idxs[ones_pos[1]], :] += (m2 * w).astype(np.float32)
                counts_16x256[idxs[c2_pos],      :] += (m_c2 * w).astype(np.float32)
                counts_16x256[idxs[c3_pos],      :] += (m_c3 * w).astype(np.float32)

def invert_key_schedule_aes128(round10_key):
    """Invert AES-128 key schedule to get the main (round 0) key.
    round10_key: array-like length 16 (bytes at positions 0..15).
    Returns main_key (16,).
    """
    # AES-128 has 11 round keys (0..10). Each round key is 16 bytes -> 44 words (4 bytes)
    # We'll reconstruct all backward from round10_key.
    Rcon = [0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36]
    SBOX = [  # forward S-box for key schedule inversion
      99,124,119,123,242,107,111,197, 48,  1,103, 43,254,215,171,118,
     202,130,201,125,250, 89, 71,240,173,212,162,175,156,164,114,192,
     183,253,147, 38, 54, 63,247,204, 52,165,229,241,113,216, 49, 21,
       4,199, 35,195, 24,150,  5,154,  7, 18,128,226,235, 39,178,117,
       9,131, 44, 26, 27,110, 90,160, 82, 59,214,179, 41,227, 47,132,
      83,209,  0,237, 32,252,177, 91,106,203,190, 57, 74, 76, 88,207,
     208,239,170,251, 67, 77, 51,133, 69,249,  2,127, 80, 60,159,168,
      81,163, 64,143,146,157, 56,245,188,182,218, 33, 16,255,243,210,
     205, 12, 19,236, 95,151, 68, 23,196,167,126, 61,100, 93, 25,115,
      96,129, 79,220, 34, 42,144,136, 70,238,184, 20,222, 94, 11,219,
     224, 50, 58, 10, 73,  6, 36, 92,194,211,172, 98,145,149,228,121,
     231,200, 55,109,141,213, 78,169,108, 86,244,234,101,122,174,  8,
     186,120, 37, 46, 28,166,180,198,232,221,116, 31, 75,189,139,138,
     112, 62,181,102, 72,  3,246, 14, 97, 53, 87,185,134,193, 29,158,
     225,248,152, 17,105,217,142,148,155, 30,135,233,206, 85, 40,223,
     140,161,137, 13,191,230, 66,104, 65,153, 45, 15,176, 84,187, 22
    ]
    def RotWord(w): return w[1:]+w[:1]
    def SubWord(w): return [SBOX[b] for b in w]
    rk = [0]*176
    rk[160:176] = list(round10_key)  # last round key bytes
    # Reconstruct previous round keys backward
    for r in range(10, 0, -1):
        # rk bytes indices for round r and r-1
        i = r*16
        prev = (r-1)*16
        # words w0..w3 at round r-1
        # w3_{r-1} = w3_r ^ w2_{r-1}
        # w2_{r-1} = w2_r ^ w1_{r-1}
        # w1_{r-1} = w1_r ^ w0_{r-1}
        # w0_{r-1} = w0_r ^ SubWord(RotWord(w3_{r-1})) ^ [Rcon[r],0,0,0]
        w0r = rk[i:i+4]
        w1r = rk[i+4:i+8]
        w2r = rk[i+8:i+12]
        w3r = rk[i+12:i+16]

        # We solve backwards by reconstructing w3_{r-1}, w2_{r-1}, w1_{r-1}, w0_{r-1}
        # Start with temp = SubWord(RotWord(w3_{r-1})) ^ [Rcon[r],0,0,0], but w3_{r-1} unknown.
        # The standard backward derivation:
        # w3_{r-1} = w3_r ^ w2_{r}
        w3m1 = [a ^ b for a,b in zip(w3r, w2r)]
        w2m1 = [a ^ b for a,b in zip(w2r, w1r)]
        w1m1 = [a ^ b for a,b in zip(w1r, w0r)]
        temp = SubWord(RotWord(w3m1))
        temp[0] ^= Rcon[r]
        w0m1 = [a ^ b for a,b in zip(w0r, temp)]

        rk[prev:prev+4] = w0m1
        rk[prev+4:prev+8] = w1m1
        rk[prev+8:prev+12] = w2m1
        rk[prev+12:prev+16] = w3m1
    main_key = rk[0:16]
    return np.array(main_key, dtype=np.uint8)

def recover_round10_key_and_counts(C_all, F_all, max_pairs=None):
    """Main loop: accumulate counts from pairs and produce Round10 key + per-column stabilization pairs.
    Returns:
      - round10_key (16,)
      - pairs_needed_per_col (list of 4 ints)
      - history_top (pairs x 16) top key guess per byte after each pair
    """
    N = C_all.shape[0]
    if max_pairs is None: max_pairs = N
    counts = np.zeros((16, 256), dtype=np.float32)
    history_top = []  # list of (16,) ints of current argmax per byte

    for t in range(max_pairs):
        C = C_all[t]; F = F_all[t]
        accumulate_counts_for_pair(C, F, counts)
        top = counts.argmax(axis=1)  # (16,)
        history_top.append(top.copy())

    history_top = np.stack(history_top, axis=0)  # (T,16)
    round10_key = counts.argmax(axis=1).astype(np.uint8)  # best per byte

    # Compute stabilization point per column: first index where all 4 bytes of the column stop changing
    pairs_needed_per_col = []
    for col in range(4):
        idxs = [r + 4*((col + r) % 4) for r in range(4)]
        col_hist = history_top[:, idxs]  # (T,4)
        # Find earliest t where each of 4 series equals its final value for all subsequent times
        final_vals = col_hist[-1, :]
        t_needed = len(history_top)  # default all
        for t in range(len(history_top)):
            if np.all(col_hist[t:, :] == final_vals[None, :]):
                t_needed = t+1  # pairs are 1-based count
                break
        pairs_needed_per_col.append(int(t_needed))

    return round10_key, pairs_needed_per_col, history_top


def recover_round10_key_and_counts_early(C_all, F_all, patience=10, max_pairs=None):
    N = C_all.shape[0]
    if max_pairs is None: max_pairs = N
    counts = np.zeros((16, 256), dtype=np.float32)
    history_top = []

    stable_streak = 0
    prev_top = None

    for t in range(max_pairs):
        C = C_all[t]; F = F_all[t]
        accumulate_counts_for_pair(C, F, counts)
        top = counts.argmax(axis=1)  # (16,)
        history_top.append(top.copy())
        if prev_top is not None and np.all(top == prev_top):
            stable_streak += 1
        else:
            stable_streak = 0
        prev_top = top.copy()
        if stable_streak >= patience:
            break

    history_top = np.stack(history_top, axis=0)
    round10_key = counts.argmax(axis=1).astype(np.uint8)

    # compute per-column stabilization retrospectively
    pairs_needed_per_col = []
    for col in range(4):
        idxs = [r + 4*((col + r) % 4) for r in range(4)]
        col_hist = history_top[:, idxs]
        final_vals = col_hist[-1, :]
        t_needed = len(history_top)
        for t in range(len(history_top)):
            if np.all(col_hist[t:, :] == final_vals[None, :]):
                t_needed = t+1
                break
        pairs_needed_per_col.append(int(t_needed))

    return round10_key, pairs_needed_per_col, history_top
