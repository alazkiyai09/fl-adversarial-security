# FL Adversarial Security (`fl-adversarial-security`)

Research and engineering platform for **federated learning attacks, defenses, secure aggregation, benchmarking, and production security monitoring**.

## Why This Repository

Secure federated learning requires both offensive and defensive testing. `fl-adversarial-security` packages attack simulation, robust aggregation, benchmark surfaces, and production-oriented defense components in one repository.

## Core Features

- Attack modules: label flipping, backdoor, model poisoning, gradient leakage, membership/property inference
- Defense modules: robust aggregation, anomaly detection, FoolsGold, SignGuard
- Secure aggregation and production privacy layers
- Benchmark runners and configuration surfaces
- API routes for attacks, defenses, benchmark, and predict flows
- Dashboard and production-serving integrations

## Project Structure

- `src/attacks/`: adversarial attack modules and utilities
- `src/defenses/`: defense mechanisms and robust aggregation logic
- `src/benchmark/`: benchmarking runner and configs
- `src/secure_aggregation/`: privacy-preserving aggregation protocol code
- `src/production/`: serving, monitoring, and deployment-facing modules
- `src/api/`: unified FastAPI service

## API Endpoints

- `POST /api/v1/attacks/simulate`
- `GET /api/v1/attacks/types`
- `POST /api/v1/defenses/evaluate`
- `GET /api/v1/defenses/types`
- `POST /api/v1/benchmark/run`
- `GET /api/v1/benchmark/results`
- `POST /api/v1/predict`
- `GET /health`
- `GET /metrics`

## Quick Start

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## SEO Keywords

federated learning security, byzantine robust aggregation, backdoor attack federated learning, label flipping attack, foolsgold defense, signguard defense, secure aggregation fl
