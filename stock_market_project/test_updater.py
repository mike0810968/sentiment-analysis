#!/usr/bin/env python3
"""
Test script for data updater functionality
Run this to verify everything works before setting up automation
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from data_updater import StockDataUpdater, NewsDataUpdater
        print("✅ Data updater modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_stock_updater():
    """Test stock data updater initialization"""
    try:
        from data_updater import StockDataUpdater
        updater = StockDataUpdater()
        print("✅ Stock updater initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Stock updater error: {e}")
        return False

def test_news_updater():
    """Test news data updater initialization"""
    try:
        from data_updater import NewsDataUpdater
        updater = NewsDataUpdater()
        print("✅ News updater initialized successfully")
        return True
    except Exception as e:
        print(f"❌ News updater error: {e}")
        return False

def test_yfinance():
    """Test Yahoo Finance connectivity"""
    try:
        import yfinance as yf
        # Test with a simple ticker
        ticker = yf.Ticker("RELIANCE.NS")
        info = ticker.info
        if info:
            print("✅ Yahoo Finance connection successful")
            return True
        else:
            print("⚠️ Yahoo Finance returned empty info")
            return False
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")
        return False

def main():
    print("🧪 Testing Data Updater Components...")
    print("=" * 50)

    tests = [
        ("Module Imports", test_imports),
        ("Stock Updater", test_stock_updater),
        ("News Updater", test_news_updater),
        ("Yahoo Finance", test_yfinance),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Ready to set up automation.")
        print("\nNext steps:")
        print("1. Get API keys (optional but recommended)")
        print("2. Run: chmod +x setup_automation.sh && ./setup_automation.sh")
        print("3. Or push to GitHub for cloud automation")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print("You can still use the dashboard with static data.")

if __name__ == "__main__":
    main()