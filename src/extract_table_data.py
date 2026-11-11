"""Extract data from output 7 directory to create comparison table."""

import os
import pandas as pd
import numpy as np
from pathlib import Path

from paths import OUTPUT_ROOT, RESULTS_DIR

base_path = OUTPUT_ROOT

results = []

# Iterate through all model directories
for model_dir in sorted(base_path.iterdir()):
    if not model_dir.is_dir():
        continue
    
    # Parse model type and layers from directory name
    if "LSTM" in model_dir.name:
        model_type = "LSTM"
        layers = int(model_dir.name.split("_")[1].replace("L", ""))
    elif "Mamba" in model_dir.name:
        model_type = "Mamba"
        layers = int(model_dir.name.split("_")[1].replace("L", ""))
    else:
        continue
    
    # Iterate through tier directories
    for tier_dir in sorted(model_dir.iterdir()):
        if not tier_dir.is_dir():
            continue
        
        all_dir = tier_dir / "All"
        if not all_dir.exists():
            continue
        
        # Check if required files exist
        params_file = all_dir / "total_params.txt"
        loss_file = all_dir / "training_loss.csv"
        metrics_file = all_dir / "basin_metrics.csv"
        
        if not (params_file.exists() and loss_file.exists() and metrics_file.exists()):
            continue
        
        try:
            # Extract tier information
            tier_dir_name = tier_dir.name
            if "Tier1" in tier_dir_name:
                tier = 1
                tier_name = "200K"
            elif "Tier2" in tier_dir_name:
                tier = 2
                tier_name = "500K"
            elif "Tier3" in tier_dir_name:
                tier = 3
                tier_name = "1M"
            else:
                continue
            
            # Extract hidden size from directory name
            # Format: H162 or H160, etc.
            import re
            h_match = re.search(r'_H(\d+)_', tier_dir_name)
            if h_match:
                hidden_size = int(h_match.group(1))
            else:
                continue
            
            # Extract parameters
            with open(params_file, 'r') as f:
                params_content = f.read()
                for line in params_content.split('\n'):
                    if 'total_params:' in line:
                        total_params = int(line.split(':')[1].strip())
                        break
                else:
                    total_params = None
            
            # Extract loss at epoch 30
            loss_df = pd.read_csv(loss_file)
            ep30_loss = loss_df[loss_df['epoch'] == 30]['loss'].values[0] if 30 in loss_df['epoch'].values else None
            
            # Extract median NSE and KGE
            metrics_df = pd.read_csv(metrics_file)
            median_nse = metrics_df['NSE'].median()
            median_kge = metrics_df['KGE'].median()
            
            results.append({
                'Depth': layers,
                'Tier': tier,
                'Model': model_type,
                'H': hidden_size,
                'Parameter': total_params,
                'Loss Ep30': ep30_loss,
                'Median NSE': median_nse,
                'Median Kge': median_kge
            })
            
        except Exception as e:
            print(f"Error processing {tier_dir}: {e}")
            continue

# Create DataFrame and sort
df = pd.DataFrame(results)
df = df.sort_values(['Model', 'Depth', 'Tier'])

# Display results
print("Extracted Data:")
print(df.to_string(index=False))

# Save to CSV for inspection
df.to_csv(RESULTS_DIR / "table_data_temp.csv", index=False)
print("\nSaved to table_data_temp.csv")
