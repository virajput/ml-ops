#!/usr/bin/env python3
"""CLI entry point to download and clean the UCI Heart Disease dataset.

Usage:
    python data/download_data.py

Downloads the raw Cleveland dataset to data/raw/heart_disease_raw.csv and
writes the cleaned version to data/processed/heart_disease_clean.csv.
"""
from heart_disease.data import main

if __name__ == "__main__":
    main()
