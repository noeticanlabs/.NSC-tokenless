# NSC NPE Version Strategy

## Version Overview

NSC Tokenless supports three NPE (Noetican Proposal Engine) versions:

| Version | Status | Features | Recommendation |
|---------|--------|----------|----------------|
| **v2.0** | Default | Beam-search synthesis, full coherence metrics, receipts | **Use for new projects** |
| **v1.0** | Stable | Coherence-aligned, production-ready | Use for stability requirements |
| **v0.1** | Deprecated | Legacy implementation | Migrate to v2.0 |

## Importing

### Default (v2.0)
```python
from nsc import NPE
npe = NPE()
```

### v1.0 Specifically
```python
from nsc import NPE_v1_0
npe = NPE_v1_0()
```

### v0.1 (Deprecated)
```python
from nsc import NPE_v0_1  # Shows deprecation warning
npe = NPE_v0_1()
```

## Migration from v0.1

v0.1 is deprecated. To migrate:

1. **Update imports**:
   ```python
   # Old
   from nsc.npe_v0_1 import NPE
   
   # New (recommended)
   from nsc import NPE  # v2.0
   
   # Or v1.0
   from nsc import NPE_v1_0
   ```

2. **Update API calls** - v2.0 has enhanced features but maintains backward compatibility for core functionality.

## Version Selection in CLI

Use `--npe-version` flag (if available):
```bash
nsc execute module.json --npe-version v2  # Default
nsc execute module.json --npe-version v1  # Use v1.0
```

## Deprecation Timeline

| Date | Change |
|------|--------|
| 2026-02-12 | v2.0 promoted to default |
| 2026-06-01 | v0.1 imports show warnings |
| 2026-12-01 | v0.1 removed from package |
