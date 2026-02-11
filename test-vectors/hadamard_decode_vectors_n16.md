# Hadamard Decode Vectors n=16

Thresholds:
tau_abs = 0.60
tau_margin = 0.10

Case A (clean):
- w = v_row1 exactly
Expect: chosen=GLYPH_ID(row1), margin >= 0.10

Case B (low absolute):
- w = [0.01,...,0.01]
Expect: chosen=REJECT(control_row_id=0)

Case C (tight margin):
- s1=0.62, s2=0.58 => margin=0.04
Expect: chosen=REJECT(control_row_id=0)
