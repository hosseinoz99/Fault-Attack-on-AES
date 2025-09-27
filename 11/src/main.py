# -*- coding: utf-8 -*-
"""Assignment 11 — DFA on AES-128 (fault at MixColumns input in Round 9)
Reads plaintexts.dat (unused but loaded for completeness), ciphertexts.dat, faultytexts.dat,
recovers Round 10 key bytes, inverts the key schedule to MAIN key, and writes:
- outputs/key.txt
- outputs/number_of_pairs.txt
- outputs/round10_key.txt
"""
import os, json, numpy as np
from .dfa_round10 import recover_round10_key_and_counts_early, invert_key_schedule_aes128

def load_dat_16xN(path):
    data = np.fromfile(path, dtype=np.uint8)
    assert data.size % 16 == 0, "Data length not divisible by 16"
    return data.reshape((-1,16))

def run(paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    P = load_dat_16xN(paths.get("plaintexts", "")) if os.path.exists(paths.get("plaintexts","")) else None
    C = load_dat_16xN(paths["ciphertexts"])
    F = load_dat_16xN(paths["faultytexts"])
    # Recover last-round key & counts
    r10, pairs_needed_per_col, hist = recover_round10_key_and_counts_early(C, F, max_pairs=None)
    main_key = invert_key_schedule_aes128(r10)

    # Write outputs
    with open(os.path.join(out_dir, "round10_key.txt"), "w") as f:
        f.write("Round10 key (hex): " + "".join(f"{b:02x}" for b in r10) + "\n")
        f.write("Round10 key (dec): " + " ".join(str(int(b)) for b in r10) + "\n")

    with open(os.path.join(out_dir, "key.txt"), "w") as f:
        f.write("MAIN key (hex): " + "".join(f"{b:02x}" for b in main_key) + "\n")
        f.write("MAIN key (dec): " + " ".join(str(int(b)) for b in main_key) + "\n")

    with open(os.path.join(out_dir, "number_of_pairs.txt"), "w") as f:
        for col, n in enumerate(pairs_needed_per_col):
            f.write(f"Column {col}: {n}\n")

    return os.path.join(out_dir, "key.txt"), os.path.join(out_dir, "number_of_pairs.txt")

if __name__ == "__main__":
    import argparse, pathlib, json
    parser = argparse.ArgumentParser(description="Assignment 11 — DFA on AES-128")
    parser.add_argument("--plaintexts", type=str, required=False, default="/mnt/data/plaintexts.dat")
    parser.add_argument("--ciphertexts", type=str, required=True)
    parser.add_argument("--faultytexts", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()
    paths = {"plaintexts": args.plaintexts, "ciphertexts": args.ciphertexts, "faultytexts": args.faultytexts}
    run(paths, args.out_dir)
