#!/bin/bash

# Script to extract median NSE and KGE from basin_metrics.csv files

cd "output 7 -- full run -- LSTM and Mamba"

echo "Extracting metrics from all models..."

# Function to extract median NSE and KGE
extract_metrics() {
    local file="$1"
    local label="$2"
    
    if [ -f "$file" ]; then
        # Extract NSE and KGE columns (indices 8 and 16 in 0-based indexing)
        # Using awk to calculate median
        nse_median=$(awk -F',' 'NR>1 {print $9}' "$file" | sort -n | awk '{a[i++]=$1} END {print a[int(i/2)]}')
        kge_median=$(awk -F',' 'NR>1 {print $17}' "$file" | sort -n | awk '{a[i++]=$1} END {print a[int(i/2)]}')
        
        echo "$label | NSE: $nse_median | KGE: $kge_median"
    else
        echo "$label | File not found"
    fi
}

# LSTM 1L
extract_metrics "LSTM_1L/Tier1_200K_Ep30_Bs128_Rho365_H162_L1/All/basin_metrics.csv" "LSTM_1L_Tier1"
extract_metrics "LSTM_1L/Tier2_500K_Ep30_Bs128_Rho365_H247_L1/All/basin_metrics.csv" "LSTM_1L_Tier2"
extract_metrics "LSTM_1L/Tier3_1M_Ep30_Bs128_Rho365_H345_L1/All/basin_metrics.csv" "LSTM_1L_Tier3"

# LSTM 2L
extract_metrics "LSTM_2L/Tier1_200K_Ep30_Bs128_Rho365_H106_L2/All/basin_metrics.csv" "LSTM_2L_Tier1"
extract_metrics "LSTM_2L/Tier2_500K_Ep30_Bs128_Rho365_H177_L2/All/basin_metrics.csv" "LSTM_2L_Tier2"
extract_metrics "LSTM_2L/Tier3_1M_Ep30_Bs128_Rho365_H248_L2/All/basin_metrics.csv" "LSTM_2L_Tier3"

# LSTM 3L
extract_metrics "LSTM_3L/Tier1_200K_Ep30_Bs128_Rho365_H92_L3/All/basin_metrics.csv" "LSTM_3L_Tier1"
extract_metrics "LSTM_3L/Tier2_500K_Ep30_Bs128_Rho365_H149_L3/All/basin_metrics.csv" "LSTM_3L_Tier2"
extract_metrics "LSTM_3L/Tier3_1M_Ep30_Bs128_Rho365_H206_L3/All/basin_metrics.csv" "LSTM_3L_Tier3"

# Mamba 1L
extract_metrics "Mamba3_1L/Tier1_Ep30_Variant-adapter_V3_head32_H160_L1_dConv4/All/basin_metrics.csv" "Mamba_1L_Tier1"
extract_metrics "Mamba3_1L/Tier2_Ep30_Variant-adapter_V3_head32_H256_L1_dConv4/All/basin_metrics.csv" "Mamba_1L_Tier2"
extract_metrics "Mamba3_1L/Tier3_Ep30_Variant-adapter_V3_head32_H368_L1_dConv4/All/basin_metrics.csv" "Mamba_1L_Tier3"

# Mamba 2L
extract_metrics "Mamba3_2L/Tier1_Ep30_Variant-adapter_V3_head32_H96_L2_dConv4/All/basin_metrics.csv" "Mamba_2L_Tier1"
extract_metrics "Mamba3_2L/Tier2_Ep30_Variant-adapter_V3_head32_H176_L2_dConv4/All/basin_metrics.csv" "Mamba_2L_Tier2"
extract_metrics "Mamba3_2L/Tier3_Ep30_Variant-adapter_V3_head32_H256_L2_dConv4/All/basin_metrics.csv" "Mamba_2L_Tier3"

# Mamba 3L
extract_metrics "Mamba3_3L/Tier1_Ep30_Variant-adapter_V3_head32_H80_L3_dConv4/All/basin_metrics.csv" "Mamba_3L_Tier1"
extract_metrics "Mamba3_3L/Tier2_Ep30_Variant-adapter_V3_head32_H144_L3_dConv4/All/basin_metrics.csv" "Mamba_3L_Tier2"
extract_metrics "Mamba3_3L/Tier3_Ep30_Variant-adapter_V3_head32_H208_L3_dConv4/All/basin_metrics.csv" "Mamba_3L_Tier3"
