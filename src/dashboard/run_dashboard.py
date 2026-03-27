"""
FL Security Dashboard - Quick Run Script
Day 29: Real-time monitoring dashboard for federated learning security

Run this to start the interactive dashboard.
"""

import subprocess
import sys
from pathlib import Path

print("="*70)
print("FL SECURITY DASHBOARD")
print("="*70)
print("\nStarting Streamlit dashboard...")

# Check if streamlit is installed
try:
    import streamlit as st
    print("✅ Streamlit found")
except ImportError:
    print("❌ Streamlit not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "-q"])
    import streamlit as st
    print("✅ Streamlit installed")

# Navigate to app directory
dashboard_dir = Path(__file__).parent / "app"
print(f"\nDashboard location: {dashboard_dir}")

# Instructions
print("\n" + "="*70)
print("DASHBOARD FEATURES")
print("="*70)
print("""
📊 Pages:
  1. Training Monitor - Real-time FL training metrics
  2. Client Analytics - Per-client performance and behavior
  3. Security Status - Attack detection and alerts
  4. Privacy Budget - DP privacy accounting
  5. Experiment Comparison - Compare different runs

🎯 Key Metrics Tracked:
  • Global model accuracy/loss per round
  • Client contribution and participation
  • Anomaly detection alerts
  • Privacy budget (ε) consumption
  • Attack success rate monitoring
  • Byzantine client identification

🔒 Security Features:
  • Real-time threat level indicator
  • Anomaly alerts with detailed info
  • Client trust scores and reputation
  • Attack detection visualization

📈 Visualizations:
  • Training curves (accuracy, loss)
  • Client participation heatmap
  • Anomaly score distributions
  • Privacy budget consumption
  • Defense method comparison
""")

print("="*70)
print("LAUNCHING DASHBOARD")
print("="*70)
print("\nDashboard will open in your browser at: http://localhost:8501")
print("\nPress Ctrl+C to stop the dashboard\n")

# Launch dashboard
try:
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    str(dashboard_dir / "main.py"),
                    "--logger.level", "info"])
except KeyboardInterrupt:
    print("\n\n✅ Dashboard stopped")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nYou can also run manually:")
    print(f"  cd {dashboard_dir.parent}")
    print(f"  streamlit run app/main.py")
