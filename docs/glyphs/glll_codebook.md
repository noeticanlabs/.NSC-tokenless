# GLLL Codebook (Normative if decode used) — NSC Tokenless v1.1

GLLL provides robust identity under noise. NSC consumes the decoded identity by mapping
glyph_id -> op_id via a registry mapping.

## 1) Codebook structure
A codebook MUST define:
- n: integer vector length
- reserved_control_rows: map control name -> row_id (REJECT required)
- rows: array of row entries:
  - row_id: integer
  - glyph_id: string
  - vector: array of length n with entries in {-1, +1}
- mapping_to_op_id: map glyph_id -> op_id (control glyphs MUST NOT map)

## 2) Canonical constraints
- All vectors must be length n and contain only -1 or +1.
- row_id values must be unique.
- glyph_id values must be unique.
- REJECT control row MUST exist and MUST NOT map to op_id.
- Recommended canonicalization: enforce vector[0] = +1 by flipping sign if needed.

## 3) Minimum separation
Hadamard-derived codebooks should satisfy:
- any two distinct non-control rows have Hamming distance n/2 (Hadamard property).
If a codebook does not meet this, it MUST declare its actual d_min.

## 4) Versioning
Codebooks SHOULD be versioned. DecodeReports should record which codebook version was used.
