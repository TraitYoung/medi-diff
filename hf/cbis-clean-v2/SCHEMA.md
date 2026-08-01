# `metadata_clean.csv` schema (CBIS_CLEAN_V2)

Column names may evolve; treat this as the documented contract for MammoGen mainline.

| Column | Type | Meaning |
|--------|------|---------|
| `src` | path | Path to source JPEG (relative or absolute on the training machine) |
| `view` | str | e.g. `MLO`, `CC` |
| `density` | str | e.g. `dense`, `scattered`, … |
| `laterality` | str | `L` / `R` when available |
| `caption` / text fields | str | Optional training captions from `generate_captions.py` |

Filters used at generation time typically include `--filter-view` and `--filter-density`.
