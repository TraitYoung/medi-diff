# Security Policy

## Supported versions

Security fixes are accepted against the default branch of [TraitYoung/medi-diff](https://github.com/TraitYoung/medi-diff) (current release line: `1.x`).

## Scope

MammoGen is a **research / education** toolchain for synthetic mammography. It is **not** a medical device and must **not** be used for clinical diagnosis or patient care.

In-scope reports include:

- Remote code execution, path traversal, or unsafe deserialization in the Gradio UI / FastAPI server
- Secrets leakage (API keys, tokens) via logs, responses, or committed files
- Prompt / file injection that can overwrite unexpected paths outside the project output dirs

Out of scope:

- Model quality, evaluation false positives/negatives, or “looks clinical”
- Denial of service via large image batches on a local GPU box
- Issues that require physical access to the machine running the tools

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

1. Email the maintainer via the address on the GitHub profile of [TraitYoung](https://github.com/TraitYoung), **or**
2. Use [GitHub Security Advisories](https://github.com/TraitYoung/medi-diff/security/advisories/new) (preferred when available).

Include: affected version / commit, reproduction steps, impact, and any suggested fix.

We aim to acknowledge reports within **7 days** and to publish a fix or mitigation for confirmed issues as soon as practical.

## Secrets and data

- Keep API keys in `.env` only (see `.env.example`). Never commit `.env`, weights, or patient-derived datasets.
- Redistribute CBIS-DDSM / TCIA derived images only under the **original data terms**; this repository does not relicense those images.
