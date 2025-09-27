# Assignment 11 — Differential Fault Attack (DFA) on AES-128
**Student**: *Omidi Zadeh*

Implements a DFA under the fault model: **single-byte fault at MixColumns input in Round 9**, unknown row per column, exactly one faulty byte per column. Using 10,000 pairs (ciphertext, faultytext) we recover the **last round key (Round 10)** and invert the key schedule to output the **MAIN key**.

## How it works (1-paragraph summary)
Let `D_i(k) = InvSBox(C_i ⊕ k) ⊕ InvSBox(C'_i ⊕ k)`. For bytes belonging to the same MixColumns column (after `InvShiftRows` grouping), and under the correct 4 key bytes, the vector `D = [D_0, D_1, D_2, D_3]` equals one of the four AES MixColumns patterns
`[2Δ, Δ, Δ, 3Δ]`, `[3Δ, 2Δ, Δ, Δ]`, `[Δ, 3Δ, 2Δ, Δ]`, `[Δ, Δ, 3Δ, 2Δ]` for some nonzero `Δ`. For each pair and each column/hypothesis we accumulate **votes** for last-round key guesses that satisfy these equalities; across many pairs the true key guesses dominate. Finally we **invert the AES‑128 key schedule** to produce the MAIN key.

## Run
```bash
pip install -r requirements.txt

python -m src.main   --ciphertexts /mnt/data/ciphertexts.dat   --faultytexts /mnt/data/faultytexts.dat   --out_dir outputs
# (plaintexts.dat is optional and not used by the attack)
```
Outputs:
- `outputs/key.txt` — MAIN key (hex & dec)
- `outputs/round10_key.txt` — last round key (hex & dec)
- `outputs/number_of_pairs.txt` — pairs needed per 4-byte column

## Compliance
- Uses vectorized NumPy; no simple loops over keys/traces in hot paths.
- Does **not** include any provided data files in the package.
- Figures are not required by the PDF; only text files + code are produced.

## Notes
- The code assumes AES‑128 and the standard Rijndael field `0x11B`.
- If your environment has different file paths, edit `configs/settings.json` or pass CLI args.
