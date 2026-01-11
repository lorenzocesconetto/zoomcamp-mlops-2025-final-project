#!/bin/bash

set -e

ENDPOINT_NAME="${ENDPOINT_NAME:-crypto-price-prediction-dev-endpoint}"
REGION="${AWS_REGION:-us-east-1}"

# Sample input with all required features
PAYLOAD='{"instances":[{"open":67500.0,"high":67800.0,"low":67200.0,"close":67650.0,"volume":1250.5,"price_change_1":0.002,"price_change_5":0.008,"price_change_15":0.015,"price_change_30":0.025,"sma_5":67400.0,"sma_10":67200.0,"sma_20":67000.0,"sma_50":66500.0,"ema_5":67450.0,"ema_10":67250.0,"ema_20":67050.0,"price_sma_5_ratio":1.0037,"price_sma_20_ratio":1.0097,"price_ema_10_ratio":1.0059,"volatility_10":250.0,"volatility_20":300.0,"price_range":0.0089,"volume_sma_10":1100.0,"volume_ratio":1.137,"volume_change":0.15,"vwap":67300.0,"price_vwap_ratio":1.0052,"rsi_14":55.0,"rsi_7":58.0,"macd":150.0,"macd_signal":120.0,"macd_histogram":30.0,"bb_upper":68000.0,"bb_lower":66500.0,"bb_width":0.0224,"bb_position":0.77,"stoch_k":65.0,"stoch_d":60.0,"williams_r":-35.0,"atr":350.0,"mfi":52.0,"hour":14,"day_of_week":2,"is_weekend":0,"hour_sin":0.866,"hour_cos":-0.5,"dow_sin":0.782,"dow_cos":-0.623,"support_level":66800.0,"resistance_level":68200.0,"support_distance":0.0126,"resistance_distance":0.0081,"bid_ask_spread":600.0,"spread_ratio":0.0089}]}'

# Create temp files that auto-clean on exit
INPUT_FILE=$(mktemp)
OUTPUT_FILE=$(mktemp)
trap "rm -f $INPUT_FILE $OUTPUT_FILE" EXIT

# Write payload to temp file
echo "$PAYLOAD" > "$INPUT_FILE"

# Invoke endpoint
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name "$ENDPOINT_NAME" \
    --region "$REGION" \
    --content-type "application/json" \
    --body "fileb://$INPUT_FILE" \
    "$OUTPUT_FILE" >/dev/null

cat "$OUTPUT_FILE"
echo ""
echo ""

# Parse and display prediction
if command -v jq &> /dev/null; then
    echo "Summary:"
    PREDICTION=$(jq -r '.predictions[0]' "$OUTPUT_FILE" 2>/dev/null || echo "N/A")
    PROBABILITIES=$(jq -r '.probabilities[0]' "$OUTPUT_FILE" 2>/dev/null || echo "N/A")

    echo "  Prediction class: $PREDICTION"
    echo "  Probabilities: $PROBABILITIES"
    echo ""
    echo "  Class meanings:"
    echo "    0 = Price drops by threshold first (bearish)"
    echo "    1 = Price rises by threshold first (bullish)"
    echo "    2 = Neither threshold hit within timeframe (neutral)"
else
    echo "(Install jq for prettier output parsing)"
fi
