---
license: cc-by-3.0
task_categories:
  - image-to-image
tags:
  - mammography
  - cbis-ddsm
  - medical-imaging
pretty_name: MammoGen CBIS-CLEAN-V2 metadata
size_categories:
  - 1K<n<10K
---

# CBIS-CLEAN-V2 (MammoGen metadata)

Cleaned metadata used by [MammoGen](https://github.com/TraitYoung/medi-diff) for training / conditioned generation.

> **Not for clinical diagnosis.**  
> Images originate from **CBIS-DDSM** on [TCIA](https://www.cancerimagingarchive.net/collection/cbis-ddsm/).  
> This card documents the **cleaning pipeline and metadata schema**. Redistribute pixel data only if you comply with TCIA / CBIS-DDSM terms (commonly CC BY 3.0 with attribution).

## What is included

| Artifact | Description |
|----------|-------------|
| `metadata_clean.csv` | Canonical table (`src`, view, density, laterality, …) after burn-in cleanup |
| `SCHEMA.md` | Column definitions |
| (optional) pointers | How to obtain JPEG sources (do not assume images are re-hosted here) |

## Cleaning pipeline (code)

```
clean_cbis.py → build_breast_masks.py → clean_training_labels.py → generate_captions.py
```

Canonical local path in the GitHub repo: `datasets/CBIS_CLEAN_V2/metadata_clean.csv`.

## Attribution

Please cite CBIS-DDSM / TCIA in any publication, and link MammoGen if you use this cleaning stack:

- CBIS-DDSM collection page (TCIA)
- Community mirror often used for JPEG packs: [`sposso/CBIS-DDSM-DATASET`](https://github.com/sposso/CBIS-DDSM-DATASET)
- MammoGen: https://github.com/TraitYoung/medi-diff

## License

Metadata documentation and cleaning scripts in MammoGen are MIT.  
Underlying mammography images remain under the **original CBIS-DDSM / TCIA license** (verify current terms before re-hosting pixels). The `cc-by-3.0` dataset license tag reflects the common CBIS-DDSM redistribution posture — **confirm before upload**.
