# Deployment Guide

This guide covers deploying the Privacy-Preserving FL Fraud Detection System in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Production Considerations](#production-considerations)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8 GB
- Storage: 20 GB
- Python: 3.10+

**Recommended**:
- CPU: 8+ cores
- RAM: 16+ GB
- Storage: 50+ GB SSD
- GPU: NVIDIA GPU with CUDA support (optional)

### Software Dependencies

```bash
# Core
Python 3.10+
pip 21.0+

# Optional but recommended
Docker 24.0+
Docker Compose 2.20+
kubectl (for Kubernetes)
```

### Environment Setup

```bash
# Clone repository
cd /path/to/privacy_preserving_fl_fraud

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

---

## Local Development

### Quick Start

```bash
# Run demo (synthetic data, 10 rounds)
./scripts/run_demo.sh
```

### Training with Real Data

**1. Prepare Data**:
```bash
# Generate synthetic data
python scripts/generate_sample_data.py \
  --n-samples 100000 \
  --fraud-rate 0.05 \
  --output ./data/raw
```

**2. Configure**:
```bash
# Edit .env file
cp .env.example .env
# Update configuration
```

**3. Train**:
```bash
# Basic training
python scripts/run_training.py

# With custom config
python scripts/run_training.py \
  fl.n_rounds=100 \
  data.n_clients=10 \
  privacy.epsilon=1.0

# With preset
python scripts/run_training.py \
  preset=privacy_high
```

### Running the API

**1. Start MLflow (optional)**:
```bash
mlflow server \
  --backend-store-uri mlruns/ \
  --host 0.0.0.0 \
  --port 5000
```

**2. Start API Server**:
```bash
# Basic
uvicorn src.serving.api:create_app \
  --host 0.0.0.0 \
  --port 8000

# With reload (development)
uvicorn src.serving.api:create_app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload

# Production (4 workers)
uvicorn src.serving.api:create_app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

**3. Test API**:
```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info

# Make prediction
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_001",
    "amount": 150.0,
    "merchant_id": 12345,
    "account_id": 67890
  }'
```

---

## Docker Deployment

### Building Images

**Server Image**:
```bash
cd deployment/docker

# Build
docker build -f Dockerfile.server -t fl-server:latest .

# With custom config
docker build \
  -f Dockerfile.server \
  --build-arg CONFIG=presets/privacy_medium \
  -t fl-server:latest .
```

**Client Image**:
```bash
docker build -f Dockerfile.client -t fl-client:latest .
```

**API Image**:
```bash
docker build -f Dockerfile.api -t fl-api:latest .
```

### Docker Compose

**Local Simulation**:
```bash
cd deployment/docker

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f server

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

**Configuration**:
```yaml
# docker-compose.yml
version: '3.8'

services:
  mlflow:
    image: fl-mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - mlruns:/mlflow

  server:
    image: fl-server:latest
    ports:
      - "8080:8080"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - NUM_CLIENTS=3
    depends_on:
      - mlflow

  client-1:
    image: fl-client:latest
    environment:
      - SERVER_ADDRESS=server:8080
      - CLIENT_ID=1
    depends_on:
      - server

  client-2:
    image: fl-client:latest
    environment:
      - SERVER_ADDRESS=server:8080
      - CLIENT_ID=2
    depends_on:
      - server

  client-3:
    image: fl-client:latest
    environment:
      - SERVER_ADDRESS=server:8080
      - CLIENT_ID=3
    depends_on:
      - server

  api:
    image: fl-api:latest
    ports:
      - "8000:8000"
    environment:
      - MODEL_STORE_PATH=/models
    volumes:
      - ./models:/models
    depends_on:
      - server

volumes:
  mlruns:
```

### Multi-Node Deployment

**Server Node**:
```bash
docker run -d \
  --name fl-server \
  -p 8080:8080 \
  -v $(pwd)/models:/models \
  -v $(pwd)/logs:/logs \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  fl-server:latest
```

**Client Nodes** (on each bank's server):
```bash
docker run -d \
  --name fl-client \
  -v /path/to/local/data:/data \
  -e SERVER_ADDRESS=server.example.com:8080 \
  -e CLIENT_ID=1 \
  fl-client:latest
```

---

## Kubernetes Deployment

### Prerequisites

```bash
# Check cluster access
kubectl cluster-info
kubectl get nodes
```

### Deploying Components

**1. Create Namespace**:
```bash
kubectl create namespace fl-fraud-detection
```

**2. Deploy MLflow**:
```bash
kubectl apply -f deployment/kubernetes/mlflow.yaml -n fl-fraud-detection
```

**3. Deploy FL Server**:
```bash
kubectl apply -f deployment/kubernetes/server.yaml -n fl-fraud-detection
```

**4. Deploy Clients** (StatefulSet for persistent identity):
```bash
kubectl apply -f deployment/kubernetes/clients.yaml -n fl-fraud-detection
```

**5. Deploy API**:
```bash
kubectl apply -f deployment/kubernetes/api.yaml -n fl-fraud-detection
```

**6. Expose Services**:
```bash
# Load balancer for API
kubectl expose deployment fraud-api \
  --type=LoadBalancer \
  --port=8000 \
  -n fl-fraud-detection

# Port forward for development
kubectl port-forward svc/fl-server 8080:8080 -n fl-fraud-detection
```

### Kubernetes Manifests

**Server Deployment**:
```yaml
# server.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fl-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fl-server
  template:
    metadata:
      labels:
        app: fl-server
    spec:
      containers:
      - name: server
        image: fl-server:latest
        ports:
        - containerPort: 8080
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow:5000"
        - name: NUM_CLIENTS
          value: "10"
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage
---
apiVersion: v1
kind: Service
metadata:
  name: fl-server
spec:
  selector:
    app: fl-server
  ports:
  - port: 8080
    targetPort: 8080
```

**Client StatefulSet**:
```yaml
# clients.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: fl-client
spec:
  serviceName: fl-client
  replicas: 5
  selector:
    matchLabels:
      app: fl-client
  template:
    metadata:
      labels:
        app: fl-client
    spec:
      containers:
      - name: client
        image: fl-client:latest
        env:
        - name: SERVER_ADDRESS
          value: "fl-server:8080"
        - name: CLIENT_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.uid
        volumeMounts:
        - name: data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

### Monitoring

**Pod Status**:
```bash
kubectl get pods -n fl-fraud-detection
kubectl logs -f deployment/fl-server -n fl-fraud-detection
```

**Scaling**:
```bash
# Scale clients
kubectl scale statefulset fl-client --replicas=10 -n fl-fraud-detection

# Scale API
kubectl scale deployment fraud-api --replicas=3 -n fl-fraud-detection
```

---

## Production Considerations

### Security

**Network Security**:
- Use TLS for all communications
- Deploy in private VPC
- Network policies for pod-to-pod communication

**Secrets Management**:
```bash
# Use Kubernetes secrets
kubectl create secret generic fl-secrets \
  --from-literal=mlflow-password=xxx \
  -n fl-fraud-detection

# Or use external secret manager (Vault, AWS Secrets Manager)
```

**Authentication**:
```python
# Add to API
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict", dependencies=[Depends(security)])
async def predict(...):
    ...
```

### High Availability

**Server**:
- Multiple replicas with leader election
- Persistent storage for model checkpoints
- Graceful shutdown handling

**Clients**:
- StatefulSet for stable client identities
- Persistent volumes for local data
- Health probes for readiness/liveness

**API**:
- Horizontal Pod Autoscaler
- Load balancing
- Session affinity (if needed)

### Monitoring

**Metrics** (Prometheus):
```yaml
# Add to deployments
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
```

**Logging** (ELK/Loki):
- Centralized log aggregation
- Structured JSON logs
- Log retention policies

**Alerting** (Grafana/Prometheus):
- CPU/Memory usage
- Prediction latency
- Error rates
- Privacy budget exhaustion

### Backup and Recovery

**Model Artifacts**:
```bash
# Regular backups
kubectl exec -n fl-fraud-detection deployment/fl-server \
  -- tar czf /backup/models-$(date +%Y%m%d).tar.gz /models

# Copy to external storage
kubectl cp \
  fl-fraud-detection/deployment/fl-server:/backup/models-*.tar.gz \
  ./backups/
```

**MLflow Database**:
- Regular database dumps
- Point-in-time recovery
- Replica for read queries

---

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```
Error: CUDA out of memory
```
**Solutions**:
- Reduce batch_size
- Use gradient accumulation
- Use smaller model
- Clear cache: `torch.cuda.empty_cache()`

**2. Slow Training**
```
Training taking too long
```
**Solutions**:
- Reduce n_rounds
- Reduce local_epochs
- Increase batch_size
- Use more powerful GPU
- Enable gradient compression

**3. Privacy Budget Exhausted**
```
Privacy budget exceeded
```
**Solutions**:
- Increase target_epsilon
- Increase noise_multiplier (paradoxically helps)
- Reduce training rounds
- Use micro-batching

**4. Connection Refused**
```
Error connecting to server
```
**Solutions**:
- Check server is running
- Verify firewall rules
- Check DNS resolution
- Verify port configuration

**5. Client Disconnected**
```
Client disconnected during training
```
**Solutions**:
- Increase timeout
- Check network stability
- Add retry logic
- Use more reliable transport

### Debug Mode

**Enable Debug Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Verbose Flower**:
```bash
flwr-server --verbose
```

**Trace Issues**:
```bash
# Server logs
kubectl logs -f deployment/fl-server -n fl-fraud-detection

# Client logs
kubectl logs -f statefulset/fl-client-0 -n fl-fraud-detection

# API logs
kubectl logs -f deployment/fraud-api -n fl-fraud-detection
```

### Performance Tuning

**Optimize Data Loading**:
```python
# Increase workers
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Increase
    pin_memory=True,  # Enable for GPU
    persistent_workers=True,  # Keep workers alive
)
```

**Optimize Communication**:
```yaml
fl:
  max_message_size: 1GB  # Flower config
  client_timeout: 600
```

**Model Optimization**:
```python
# Compile model (PyTorch 2.0+)
model = torch.compile(model)

# Use half precision
model = model.half()
```

---

## Maintenance

### Updates

**Rolling Update** (Kubernetes):
```bash
kubectl set image deployment/fl-server \
  server=fl-server:v2.0 \
  -n fl-fraud-detection
```

**Blue-Green Deployment**:
1. Deploy new version alongside old
2. Test new version
3. Switch traffic
4. Decommission old version

### Scaling

**Horizontal Scaling**:
```bash
# Auto-scaling
kubectl autoscale deployment fraud-api \
  --min=2 --max=10 \
  --cpu-percent=70 \
  -n fl-fraud-detection
```

**Vertical Scaling**:
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

---

## Support

For issues and questions:
- GitHub Issues: [github.com/yourusername/privacy-preserving-fl-fraud/issues]
- Documentation: [docs/]
- Email: your.email@example.com
