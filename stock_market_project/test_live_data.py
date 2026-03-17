#!/usr/bin/env python3
"""
Test script to verify live data fetching for different timeframes
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live_data import live_stock_data
import pandas as pd

def test_timeframe_data():
    """Test that different timeframes return appropriate data ranges"""
    # Use the global live_stock_data instance

    # Test different periods
    test_cases = [
        ('RELIANCE', '1d', '1m'),
        ('RELIANCE', '5d', '1h'),
        ('RELIANCE', '1mo', '1d'),
        ('RELIANCE', '1y', '1d'),
    ]

    for ticker, period, interval in test_cases:
        print(f"\nTesting {ticker} - {period} - {interval}")
        try:
            df = live_stock_data.get_live_data(ticker, period, interval)
            if not df.empty:
                print(f"  Data points: {len(df)}")
                print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
                print(f"  Columns: {list(df.columns)}")

                # Check if data is within expected timeframe
                end_date = pd.Timestamp.now(tz='Asia/Kolkata')
                if period == '1d':
                    expected_start = end_date - pd.Timedelta(days=1)
                elif period == '5d':
                    expected_start = end_date - pd.Timedelta(days=5)
                elif period == '1mo':
                    expected_start = end_date - pd.DateOffset(months=1)
                elif period == '1y':
                    expected_start = end_date - pd.DateOffset(years=1)
                else:
                    expected_start = df['Date'].min()

                actual_start = df['Date'].min()
                if actual_start >= expected_start - pd.Timedelta(days=1):  # Allow 1 day tolerance
                    print("  ✓ Data range matches expected timeframe")
                else:
                    print(f"  ✗ Data range doesn't match - expected start: {expected_start}, actual: {actual_start}")
            else:
                print("  ✗ No data returned")
        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    test_timeframe_data()