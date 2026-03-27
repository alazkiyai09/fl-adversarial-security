# fl-adversarial-security

Attack, defense, benchmark, and secure-production FL research split out of `fl-security-research`.

This repo consolidates the original security-focused projects into one structure:

- `src/attacks/` for poisoning and privacy attacks
- `src/defenses/` for robust aggregation, anomaly detection, FoolsGold, and SignGuard
- `src/secure_aggregation/` for encrypted aggregation research
- `src/benchmark/` for attack-vs-defense evaluation
- `src/production/` for the production secure FL stack
- `src/dashboard/` for monitoring and visualization
- `src/api/` for the unified FastAPI surface

Preserved project implementations remain in place under the new layout, especially in `legacy/` folders. Day 30 capstone paper material is carried in `docs/`.

## Legacy Sources Merged

- `label_flipping_attack`
- `backdoor_attack_fl`
- `model_poisoning_fl`
- `gradient_leakage_attack`
- `membership_inference_attack`
- `property_inference_attack`
- `byzantine_robust_fl`
- `fl_anomaly_detection`
- `foolsgold_defense`
- `signguard_defense`
- `signguard`
- `fl_defense_benchmark`
- `secure_aggregation_fl`
- `privacy_preserving_fl_fraud`
- `fl_security_dashboard`

## Quick Start

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## Smoke Test

```bash
pytest -q tests/test_api.py
```
