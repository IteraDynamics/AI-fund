"""
Sandboxed Python code execution for QuantAnalyst agents.
Runs code in a subprocess with a timeout and restricted builtins.
Captures stdout/stderr as the result.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional


# Packages available inside the sandbox
ALLOWED_IMPORTS = {
    "numpy", "pandas", "scipy", "sklearn", "statsmodels",
    "yfinance", "matplotlib", "math", "statistics", "datetime",
    "json", "csv", "itertools", "functools", "collections",
}

SANDBOX_TIMEOUT_S = 30  # hard kill after 30 seconds


def execute_python(
    code: str,
    timeout: int = SANDBOX_TIMEOUT_S,
    extra_context: Optional[dict] = None,
) -> dict:
    """
    Execute a Python code string in a sandboxed subprocess.

    Returns:
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "return_value": None  # future: capture last-expression value
        }
    """
    # Prepend safety header that restricts dangerous operations
    safety_header = textwrap.dedent("""\
        import sys, os
        # Block dangerous stdlib modules
        _BLOCKED = {'subprocess', 'socket', 'shutil', 'ctypes', 'importlib'}
        _real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
        def _safe_import(name, *args, **kwargs):
            if name in _BLOCKED:
                raise ImportError(f"Module '{name}' is not allowed in sandbox")
            return _real_import(name, *args, **kwargs)
        if hasattr(__builtins__, '__import__'):
            __builtins__.__import__ = _safe_import
        # Suppress plot display
        import matplotlib
        matplotlib.use('Agg')
    """)

    full_code = safety_header + "\n" + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:4000],  # cap output for LLM context
            "stderr": result.stderr[:2000],
            "return_value": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Code execution timed out after {timeout}s",
            "return_value": None,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "return_value": None,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_backtest(strategy_code: str) -> dict:
    """
    Convenience wrapper: execute strategy code and return results.
    The strategy code should print its results (Sharpe, max DD, etc.) to stdout.
    """
    return execute_python(strategy_code)


def run_factor_model(factor_code: str) -> dict:
    """Run a factor model construction script."""
    return execute_python(factor_code)
