#!/bin/bash
# Quick Test Script - Run Linear Regression Project
# This script demonstrates how to run any ML project

echo "========================================"
echo "ML ALGORITHMS PORTFOLIO - QUICK TEST"
echo "========================================"
echo ""

echo "Testing: Linear Regression Project"
echo ""

cd 01_linear_regression

echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Running Linear Regression..."
echo ""

python main.py

echo ""
echo "========================================"
echo "Test completed!"
echo "Check the visualizations folder for results."
echo "========================================"
echo ""

cd ..
