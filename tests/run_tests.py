#!/usr/bin/env python3
"""Zero-dependency test runner.

The project's runtime dependency is numpy alone, and pytest is not guaranteed to
be installed. This runner implements the small slice of the pytest API the suite
actually uses — `approx`, `mark.parametrize`, `raises` — by injecting a shim
module before the test modules import it.

The tests are written as ordinary pytest tests, so `pytest tests/` works too
wherever pytest IS available. This script is the fallback, not a replacement.

Usage:
    python tests/run_tests.py            # run everything
    python tests/run_tests.py rules      # only modules matching "rules"
    python tests/run_tests.py -v         # show every test name
"""

from __future__ import annotations

import importlib.util
import math
import sys
import traceback
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# Minimal pytest shim
# --------------------------------------------------------------------------

class _Approx:
    """Tolerant float comparison, mirroring pytest.approx semantics."""

    def __init__(self, expected: Any, rel: float = 1e-6, abs_: float = 1e-12) -> None:
        self.expected = expected
        self.rel = rel
        self.abs = abs_

    def _close(self, a: float, b: float) -> bool:
        if isinstance(b, bool) or isinstance(a, bool):
            return a == b
        try:
            return math.isclose(float(a), float(b), rel_tol=self.rel, abs_tol=max(self.abs, 1e-9))
        except (TypeError, ValueError):
            return a == b

    def __eq__(self, other: Any) -> bool:
        if isinstance(self.expected, (list, tuple)):
            if not isinstance(other, (list, tuple)) or len(other) != len(self.expected):
                return False
            return all(self._close(o, e) for o, e in zip(other, self.expected))
        if isinstance(self.expected, dict):
            if not isinstance(other, dict) or set(other) != set(self.expected):
                return False
            return all(self._close(other[k], self.expected[k]) for k in self.expected)
        return self._close(other, self.expected)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return f"approx({self.expected!r}, rel={self.rel})"


def _approx(expected: Any, rel: float | None = None, abs: float | None = None) -> _Approx:
    return _Approx(expected, rel if rel is not None else 1e-6, abs if abs is not None else 1e-12)


class _RaisesContext:
    def __init__(self, expected, match: str | None = None) -> None:
        self.expected = expected
        self.match = match
        self.value: BaseException | None = None

    def __enter__(self) -> "_RaisesContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            raise AssertionError(f"expected {self.expected} to be raised, nothing was")
        if not issubclass(exc_type, self.expected):
            return False
        if self.match is not None:
            import re

            if not re.search(self.match, str(exc)):
                raise AssertionError(
                    f"exception message {str(exc)!r} does not match {self.match!r}"
                )
        self.value = exc
        return True


def _raises(expected, match: str | None = None) -> _RaisesContext:
    return _RaisesContext(expected, match)


class _Skipped(Exception):
    """Raised by pytest.skip()."""


def _skip(reason: str = "") -> None:
    raise _Skipped(reason)


def _fail(reason: str = "") -> None:
    raise AssertionError(reason)


class _Mark:
    """Implements @pytest.mark.parametrize and tolerates other marks."""

    @staticmethod
    def parametrize(argnames: str | List[str], argvalues: List[Any], **kwargs):
        if isinstance(argnames, str):
            names = [n.strip() for n in argnames.split(",") if n.strip()]
        else:
            names = list(argnames)

        def decorator(func: Callable) -> Callable:
            cases: List[Tuple[Dict[str, Any], str]] = []
            for values in argvalues:
                if len(names) == 1:
                    bound = {names[0]: values}
                    label = repr(values)
                else:
                    seq = values if isinstance(values, (list, tuple)) else (values,)
                    if len(seq) != len(names):
                        raise ValueError(
                            f"parametrize arity mismatch for {func.__name__}: "
                            f"{names} vs {seq!r}"
                        )
                    bound = dict(zip(names, seq))
                    label = "-".join(repr(v) for v in seq)
                cases.append((bound, label))

            existing = getattr(func, "_param_cases", None)
            if existing:
                # Stacked parametrize decorators: take the cross product.
                combined: List[Tuple[Dict[str, Any], str]] = []
                for outer, olabel in cases:
                    for inner, ilabel in existing:
                        merged = dict(inner)
                        merged.update(outer)
                        combined.append((merged, f"{olabel}-{ilabel}"))
                func._param_cases = combined
            else:
                func._param_cases = cases
            return func

        return decorator

    def __getattr__(self, name: str):
        # Unknown marks (skipif, slow, ...) become no-op decorators.
        def passthrough(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            def decorator(func):
                return func

            return decorator

        return passthrough


def _install_shim() -> None:
    if "pytest" in sys.modules:
        return
    try:
        import pytest  # noqa: F401  (real pytest available, prefer it)

        return
    except ImportError:
        pass

    shim = types.ModuleType("pytest")
    shim.approx = _approx
    shim.raises = _raises
    shim.skip = _skip
    shim.fail = _fail
    shim.mark = _Mark()
    shim.Skipped = _Skipped
    sys.modules["pytest"] = shim


# --------------------------------------------------------------------------
# Discovery and execution
# --------------------------------------------------------------------------

def _load_module(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def main(argv: List[str]) -> int:
    verbose = "-v" in argv or "--verbose" in argv
    filters = [a for a in argv if not a.startswith("-")]

    _install_shim()

    test_dir = Path(__file__).resolve().parent
    files = sorted(test_dir.glob("test_*.py"))
    if filters:
        files = [f for f in files if any(k in f.stem for k in filters)]

    if not files:
        print("no test modules matched")
        return 1

    passed = failed = skipped = 0
    failures: List[Tuple[str, str]] = []

    for path in files:
        try:
            module = _load_module(path)
        except Exception:
            failed += 1
            failures.append((path.stem, traceback.format_exc()))
            print(f"\n{path.stem}: IMPORT ERROR")
            continue

        tests = [
            (name, obj)
            for name, obj in vars(module).items()
            if name.startswith("test_") and callable(obj)
        ]
        if not tests:
            continue

        print(f"\n{path.stem}  ({len(tests)} test functions)")

        for name, func in tests:
            cases = getattr(func, "_param_cases", [({}, "")])
            for kwargs, label in cases:
                display = f"{name}[{label}]" if label else name
                try:
                    func(**kwargs)
                    passed += 1
                    if verbose:
                        print(f"  PASS  {display}")
                except _Skipped as exc:
                    skipped += 1
                    if verbose:
                        print(f"  SKIP  {display}: {exc}")
                except Exception:
                    failed += 1
                    failures.append((f"{path.stem}::{display}", traceback.format_exc()))
                    print(f"  FAIL  {display}")

        if not verbose:
            print(f"  {len(tests)} functions run")

    if failures:
        print("\n" + "=" * 72)
        print("FAILURES")
        print("=" * 72)
        for name, tb in failures:
            print(f"\n--- {name} ---")
            print(tb.rstrip())

    print("\n" + "=" * 72)
    print(f"passed={passed}  failed={failed}  skipped={skipped}")
    print("=" * 72)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
