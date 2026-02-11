# GLLL Decode (Normative if decode used) — NSC Tokenless v1.1

Decode maps observed vectors to glyph identity or REJECT.

## 1) Inputs
- codebook with vectors v_row ∈ {-1,+1}^n
- observed vector w ∈ ℝ^n or {-1,+1}^n
- thresholds tau_abs, tau_margin
- top-k value K (>= 1)

## 2) Scoring
Define correlation score:
s(row) = (w · v_row) / n

This yields s ∈ [-1,1] for ±1 vectors and bounded real inputs.

## 3) Decision rule
Let best row r1 have score s1, second-best r2 score s2.
margin = s1 - s2

If (s1 < tau_abs) OR (margin < tau_margin):
- output REJECT control outcome
Else:
- output GLYPH_ID corresponding to r1

## 4) Unique decoding bound (Hadamard rows)
If codebook is Hadamard-derived with minimum Hamming distance:
d_min = n/2

Hard-decision unique decoding guarantee:
t ≤ floor((d_min - 1)/2) = floor((n - 2)/4)

This is a strict guarantee. Soft decoding may succeed beyond this but is not guaranteed.

## 5) DecodeReport
A decode attempt MUST emit a DecodeReport containing:
- n
- chosen kind GLYPH_ID or REJECT with control_row_id
- topk candidates with glyph_id and score
- thresholds used
- margin value
