# Canonical JSON Hash Vectors (canonical-json-v1)

These vectors define canonical payload strings; compute sha256 over UTF-8 bytes.

Vector 1: minimal object
Payload:
{"a":1,"b":[2,3],"c":{"d":4}}
Rule: keys sorted, no whitespace.

Vector 2: key ordering
Input objects with keys out of order must serialize to:
{"a":0,"m":1,"z":2}

Vector 3: prohibited NaN/Inf
Payload containing NaN/Inf MUST be rejected before hashing.
