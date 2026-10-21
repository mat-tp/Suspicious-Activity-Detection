"""Reports the docstring coverage of the package (modules, classes and public functions).

    python scripts/check_docstrings.py
    python scripts/check_docstrings.py --min 0.9
    python scripts/check_docstrings.py --list-all

Public means the name has no leading underscore. ``__init__`` counts only when its class has no
docstring. Other dunder methods and one-line property bodies are not counted.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

# This is the package whose docstrings are checked.
PACKAGE = Path(__file__).resolve().parent.parent / "source" / "activity_detection"


def _is_trivial(node: ast.AST) -> bool:
    """Single-statement bodies (return/pass/raise/...) of at most two lines are not counted."""
    body = getattr(node, "body", [])
    return len(body) == 1 and (body[0].end_lineno - body[0].lineno) <= 1 and isinstance(
        body[0], (ast.Return, ast.Pass, ast.Raise, ast.Expr))


def walk(path: Path) -> Iterator[Tuple[str, str, bool]]:
    """Yield (qualified name, kind, has_docstring) for every item that needs one."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rel = path.relative_to(PACKAGE.parent)
    yield f"{rel}", "module", ast.get_docstring(tree) is not None

    def visit(node: ast.AST, prefix: str) -> Iterator[Tuple[str, str, bool]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                if not child.name.startswith("_"):
                    yield f"{rel}:{prefix}{child.name}", "class", ast.get_docstring(child) is not None
                    yield from visit(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = child.name
                public = not name.startswith("_") or name == "__init__"
                # Arguments of a constructor may be documented in the class docstring.
                if name == "__init__" and isinstance(node, ast.ClassDef) and ast.get_docstring(node):
                    continue
                if public and not _is_trivial(child):
                    yield f"{rel}:{prefix}{name}", "function", ast.get_docstring(child) is not None

    yield from visit(tree, "")


def collect() -> List[Tuple[str, str, bool]]:
    items: List[Tuple[str, str, bool]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "__init__.py" and path.stat().st_size == 0:
            continue
        items.extend(walk(path))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min", type=float, default=0.0, help="fail (exit 1) if coverage is below this fraction")
    ap.add_argument("--list-all", action="store_true")
    args = ap.parse_args()
    items = collect()
    missing = [i for i in items if not i[2]]
    for kind in ("module", "class", "function"):
        sub = [i for i in items if i[1] == kind]
        got = sum(1 for i in sub if i[2])
        print(f"{kind:<9} {got:>4}/{len(sub):<4} documented ({got / max(len(sub), 1):.0%})")
    for name, kind, _ in (items if args.list_all else missing):
        print(f"  {'ok     ' if args.list_all and _ else 'MISSING'} {kind:<8} {name}")
    cov = 1 - len(missing) / max(len(items), 1)
    print(f"\nOverall: {cov:.1%}")
    return 0 if cov >= args.min else 1


if __name__ == "__main__":
    sys.exit(main())
