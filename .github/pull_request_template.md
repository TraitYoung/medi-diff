## Summary

<!-- What and why -->

## Test plan

- [ ] `pytest -m "not heavy"`
- [ ] `ruff check scripts/core scripts/assistant/tuning_state.py scripts/tests`
- [ ] (if UI/API) `python3 scripts/tools/verify_ui_wiring.py`
- [ ] (if generation/eval) describe GPU checks run locally

## Checklist

- [ ] No weights / datasets / secrets committed
- [ ] Docs updated when paths or defaults change
- [ ] Still clearly **not for clinical diagnosis**
