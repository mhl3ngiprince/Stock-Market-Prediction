"""Backwards-compatible entry point - delegates to the pipeline reporter.

Prefer:  python pipeline.py --report [--export]
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pipeline import report

if __name__ == "__main__":
    export = "--export" in sys.argv
    report(export=export)
