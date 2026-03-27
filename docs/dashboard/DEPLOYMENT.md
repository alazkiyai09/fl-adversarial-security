# Deployment Guide - FL Security Dashboard

This guide covers deployment options for the FL Security Dashboard.

## Table of Contents

1. [Local Development](#local-development)
2. [Production Deployment (Docker)](#production-deployment-docker)
3. [Docker Compose (with Redis)](#docker-compose-with-redis)
4. [Cloud Deployment](#cloud-deployment)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites

- Python 3.11 or higher
- pip or pip3
- Git

### Installation

```bash
# Clone or navigate to project
cd /home/ubuntu/30Days_Project/fl_security_dashboard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app/main.py
```

The dashboard will be available at `http://localhost:8501`

### Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
STREAMLIT_PORT=8501
STREAMLIT_ADDRESS=localhost
DEMO_MODE=true
DEFAULT_REFRESH_RATE=2000
```

---

## Production Deployment (Docker)

### Building the Docker Image

```bash
cd /home/ubuntu/30Days_Project/fl_security_dashboard
docker build -t fl-security-dashboard -f deployment/Dockerfile .
```

### Running the Container

```bash
docker run -d \
  --name fl-dashboard \
  -p 8501:8501 \
  -e DEMO_MODE=true \
  -v $(pwd)/data:/app/data \
  fl-security-dashboard
```

### Docker Options

| Option | Description |
|--------|-------------|
| `-d` | Run in detached mode (background) |
| `-p 8501:8501` | Port mapping (host:container) |
| `-e DEMO_MODE=true` | Set environment variables |
| `-v $(pwd)/data:/app/data` | Mount data directory |
| `--restart always` | Auto-restart on failure |

### Docker Compose

```bash
cd /home/ubuntu/30Days_Project/fl_security_dashboard
docker-compose -f deployment/docker-compose.yml up -d
```

This starts:
- Dashboard on port 8501
- Redis on port 6379

---

## Docker Compose (with Redis)

### Full Stack Deployment

The `deployment/docker-compose.yml` file defines:

```yaml
services:
  dashboard:
    build: ..
    ports:
      - "8501:8501"
    environment:
      - DEMO_MODE=true
      - REDIS_HOST=redis
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f dashboard

# Stop all services
docker-compose down

# Restart specific service
docker-compose restart dashboard
```

---

## Cloud Deployment

### AWS Deployment

#### Option 1: EC2 + Docker

1. Launch EC2 instance (Ubuntu 22.04)
2. Install Docker:
   ```bash
   sudo apt-get update
   sudo apt-get install docker.io docker-compose
   ```
3. Clone repository
4. Run with Docker Compose

#### Option 2: ECS (Elastic Container Service)

1. Create ECR repository
2. Push Docker image:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
   docker tag fl-security-dashboard:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/fl-dashboard:latest
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/fl-dashboard:latest
   ```
3. Create ECS task definition
4. Run task

### Google Cloud Platform

#### Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/fl-dashboard

# Deploy to Cloud Run
gcloud run deploy fl-dashboard \
  --image gcr.io/PROJECT_ID/fl-dashboard \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# Create resource group
az group create --name fl-dashboard-rg --location eastus

# Deploy container
az container create \
  --resource-group fl-dashboard-rg \
  --name fl-dashboard \
  --image fl-security-dashboard:latest \
  --dns-name-label fl-dashboard-unique \
  --ports 8501
```

---

## Performance Tuning

### Scaling Considerations

#### Number of Clients

- **Demo**: 10-50 clients (recommended)
- **Production**: 100+ clients (requires optimization)

#### Refresh Rate

Adjust in `.streamlit/config.toml` or via UI:

```toml
[client]
maxMessageSize = 500  # MB
```

Lower refresh rates = better performance but less real-time feel

#### Caching

Data fetchers use `@st.cache_data(ttl=1)` for 1-second caching. Adjust in:

```python
# app/components/data_fetchers.py
@st.cache_data(ttl=1)  # Change TTL here
```

### Memory Optimization

For large numbers of clients:

1. **Reduce history size** in `MetricsCollector`:
   ```python
   mc = MetricsCollector(max_history=100)  # Default 1000
   ```

2. **Use pagination** in client analytics:
   ```python
   # Show 50 clients at a time
   page = st.number_input("Page", 1, (num_clients // 50) + 1)
   ```

3. **Aggregate data** for charts:
   ```python
   # Show summary instead of individual clients
   summary = mc.get_summary_statistics()
   ```

---

## Troubleshooting

### Dashboard Won't Start

**Problem**: `ModuleNotFoundError`

**Solution**:
```bash
pip install -r requirements.txt
```

---

### Port Already in Use

**Problem**: `Address already in use`

**Solution**:
```bash
# Find process using port 8501
lsof -i :8501

# Kill the process
kill -9 <PID>

# Or use different port
streamlit run app/main.py --server.port 8502
```

---

### Demo Data Not Loading

**Problem**: No demo scenarios available

**Solution**:
```bash
# Generate demo data
python3 backend/generate_demo_simple.py
```

---

### High Memory Usage

**Problem**: Dashboard consuming too much RAM

**Solution**:
1. Reduce `max_history` in MetricsCollector
2. Lower number of concurrent clients
3. Increase refresh rate (reduce frequency)

---

### Redis Connection Failed

**Problem**: Cannot connect to Redis

**Solution**:
```bash
# Check if Redis is running
docker ps | grep redis

# Start Redis if not running
docker-compose up -d redis

# Or run without Redis (demo mode)
export DEMO_MODE=true
```

---

### WebSocket Connection Issues

**Problem**: Real-time updates not working

**Solution**:
1. Check WebSocket server is running:
   ```bash
   python3 backend/websocket/server.py
   ```

2. Verify firewall allows port 8765

3. Use demo mode (polling) instead:
   ```bash
   export DEMO_MODE=true
   ```

---

## Security Best Practices

1. **Don't expose Redis** publicly
2. **Use HTTPS** in production (via nginx/traefik)
3. **Add authentication** for multi-user deployments
4. **Sanitize user inputs** in configuration editor
5. **Use secrets management** for sensitive config

---

## Monitoring

### Health Checks

```bash
# Check if Streamlit is running
curl http://localhost:8501/_stcore/health

# Docker health status
docker ps
docker inspect fl-dashboard | grep Health -A 5
```

### Logs

```bash
# Streamlit logs
docker logs -f fl-dashboard

# Redis logs
docker logs -f fl_dashboard_redis
```

---

## Backup and Recovery

### Export Data

```bash
# Backup demo data
cp -r data/demo_scenarios data/backup/

# Export experiment results
python3 -c "
from app.utils.session import get_metrics_collector, save_experiment_to_history
mc = get_metrics_collector()
result = mc.export_experiment_result('backup', 'Manual Backup')
print(result.model_dump_json())
"
```

---

## Updates and Maintenance

### Updating the Dashboard

```bash
# Pull latest changes
git pull origin main

# Rebuild Docker image
docker build -t fl-security-dashboard:latest -f deployment/Dockerfile .

# Restart containers
docker-compose up -d --force-recreate
```

### Database Migration (Future)

If using a database for persistence:

```bash
# Run migrations (example using Alembic)
alembic upgrade head
```

---

## Support

For issues:
1. Check [Troubleshooting](#troubleshooting)
2. Review GitHub Issues
3. Contact: [your-email@example.com]
