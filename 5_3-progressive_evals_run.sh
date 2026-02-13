#!/bin/bash

# Date range
START_DATE="2026-02-11"
END_DATE="2026-02-11"

# Strategies
STRATEGIES=("cumulative" "penalized" "reseted")
# STRATEGIES=("reseted")

# Convert dates to seconds since epoch
start_ts=$(date -d "$START_DATE" +%s)
end_ts=$(date -d "$END_DATE" +%s)

# Loop over dates
current_ts=$start_ts
while [ "$current_ts" -le "$end_ts" ]; do
    current_date=$(date -d "@$current_ts" +%Y-%m-%d)

    # Inner loop over strategies
    for strategy in "${STRATEGIES[@]}"; do
        echo "Running for date=$current_date strategy=$strategy"
        python 5_3-progressive_experiment_results.py \
            "$current_date" \
            --matrix_strategy "$strategy" \
            --validation_run 600
    done

    # Move to next day (add 1 day = 86400 seconds)
    current_ts=$((current_ts + 86400))
done
