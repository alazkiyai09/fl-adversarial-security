#!/bin/bash

# Script to generate final report from existing results

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

RESULTS_DIR=${RESULTS_DIR:-"results"}
OUTPUT_REPORT=${OUTPUT_REPORT:-"$RESULTS_DIR/final_report.md"}

echo "=========================================="
echo "Generating FL Defense Benchmark Report"
echo "=========================================="
echo "Results Directory: $RESULTS_DIR"
echo "Output Report: $OUTPUT_REPORT"
echo "=========================================="

python -c "
import sys
sys.path.insert(0, 'src')
from visualization import load_results_json, generate_markdown_report, generate_all_tables
import yaml

# Load results
results_path = '$RESULTS_DIR/full_results.json'
try:
    with open(results_path, 'r') as f:
        import json
        results = json.load(f)
except FileNotFoundError:
    print(f'Error: Results not found at {results_path}')
    print('Please run experiments first using scripts/run_full_sweep.sh')
    sys.exit(1)

# Generate report
generate_markdown_report(
    results,
    results.get('config', {}),
    '$OUTPUT_REPORT'
)

# Generate LaTeX tables
generate_all_tables(results, '$RESULTS_DIR/tables')

print('Report generated successfully!')
print(f'Markdown report: $OUTPUT_REPORT')
print(f'LaTeX tables: $RESULTS_DIR/tables/')
"

echo ""
echo "Report generation completed!"
echo "Open $OUTPUT_REPORT to view results"
