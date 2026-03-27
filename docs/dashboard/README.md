# FL Security Dashboard

Real-time monitoring and visualization dashboard for Federated Learning security. Designed for researchers, practitioners, and academic presentations.

## Features

- **Training Monitor**: Real-time loss/accuracy curves, convergence tracking, client status
- **Client Analytics**: Per-client metrics, reputation scores, anomaly detection
- **Security Status**: Attack detection alerts, defense actions, threat level monitoring
- **Privacy Budget**: DP epsilon tracking, secure aggregation status
- **Experiment Comparison**: Side-by-side comparison of different configurations

## Tech Stack

- **Frontend**: Streamlit, Plotly
- **Backend**: Python, asyncio (planned WebSocket support)
- **Caching**: Redis (planned)
- **Deployment**: Docker, Docker Compose

## Installation

```bash
# Clone repository
cd /home/ubuntu/30Days_Project/fl_security_dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the dashboard
streamlit run app/main.py
```

The dashboard will open at `http://localhost:8501`

## Usage

### Demo Mode

1. Open the dashboard
2. Click "Start Training" in the Training Monitor page
3. Watch real-time metrics update
4. Navigate to other pages to see:
   - Client Analytics: Per-client reputation and anomaly scores
   - Security Status: Attack detection alerts
   - Privacy Budget: Epsilon tracking if DP is enabled

### Injecting Attacks (Demo)

1. Go to the Security Status page
2. Expand "Inject Attack (Demo Mode)"
3. Select attack type (label flipping, backdoor, Byzantine, poisoning)
4. Configure attack parameters
5. Click "Inject Attack"

### Configuration

Use the Configuration Editor on any page to adjust:
- FL parameters (rounds, clients, learning rate)
- Attack configuration
- Defense mechanism (SignGuard, Krum, FoolsGold, etc.)
- Privacy parameters (DP epsilon, noise multiplier)

## Project Structure

```
fl_security_dashboard/
├── app/
│   ├── main.py                  # Streamlit entry point
│   ├── pages/                   # Multi-page components
│   │   ├── training_monitor.py
│   │   ├── client_analytics.py
│   │   ├── security_status.py
│   │   ├── privacy_budget.py
│   │   └── experiment_comparison.py
│   ├── components/              # Reusable UI components
│   │   ├── charts.py            # Plotly charts
│   │   ├── metrics.py           # Metric cards, badges
│   │   ├── controls.py          # Interactive controls
│   │   └── data_fetchers.py     # Data fetching with caching
│   └── utils/                   # Utilities
│       ├── config.py            # Configuration management
│       └── session.py           # Streamlit session state
├── core/                        # Business logic
│   ├── data_models.py           # Pydantic models
│   ├── attack_engine.py         # Attack simulation
│   ├── defense_engine.py        # Defense mechanisms
│   └── metrics_collector.py     # Metrics aggregation
├── data/
│   └── demo_scenarios/          # Pre-recorded demo data
├── tests/                       # Unit tests
├── deployment/                  # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
└── requirements.txt
```

## Architecture

### Data Flow

1. **Training Loop** (simulated or real)
   - Generates client updates
   - Applies attacks (if configured)
   - Runs defense mechanism
   - Aggregates updates

2. **Metrics Collection**
   - `MetricsCollector` aggregates metrics per round
   - Maintains training history
   - Tracks security events

3. **Dashboard Updates**
   - Data fetchers retrieve metrics from collector
   - Cached with 1-second TTL
   - Auto-refresh based on configured rate

### Key Components

- **`MetricsCollector`**: Central state manager for all metrics
- **`AttackEngine`**: Simulates various attack types
- **`DefenseEngine`**: Implements SignGuard, Krum, FoolsGold, etc.
- **Chart Components**: Reusable Plotly visualizations
- **Session Management**: Streamlit session state utilities

## Security Considerations

This dashboard is designed for:
- **Educational purposes**: Understanding FL security threats
- **Research**: Comparing defense mechanisms
- **Presentations**: Visual demonstrations for academic/industry talks

**Important**: This is a simulation/monitoring tool. Do not use for production security monitoring without proper hardening.

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Pages

1. Create file in `app/pages/`
2. Define `show_page()` function
3. Import in `app/main.py`

### Adding New Charts

1. Add function to `app/components/charts.py`
2. Use `LAYOUT_TEMPLATE` for consistency
3. Return `go.Figure` object

## Deployment

### Docker

```bash
# Build image
docker build -t fl-security-dashboard .

# Run container
docker run -p 8501:8501 fl-security-dashboard
```

### Docker Compose (with Redis)

```bash
docker-compose up -d
```

## Contributing

This is part of a 30-day FL security project. Contributions welcome:
- Additional visualizations
- New attack/defense mechanisms
- Performance optimizations
- Bug fixes

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built as part of a federated learning security research portfolio.
Designed for PhD applications and AI Engineer roles.

## Citation

If you use this dashboard in your research, please cite:

```
FL Security Dashboard: Real-time Monitoring and Visualization
for Federated Learning Security
```

## Contact

For questions, issues, or collaboration opportunities, please open an issue on the repository.
