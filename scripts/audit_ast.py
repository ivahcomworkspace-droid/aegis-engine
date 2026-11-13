#!/usr/bin/env python3
"""
AST Regression Auditor for Aegis Credit Risk Engine.

Scans core/, api/, schemas/, and services/ to enforce AGENT_RULES.md:
1. Outdated Beneish weights (4.037 or 0.0327).
2. Hardcoded 100_000 or 100000 EAD floor clamps in api/.
3. Hardcoded solver_converged = True or max(v_a, clamps in core/merton.py.
4. Generic except Exception returning HTTP status_code=422 in api/.
5. External rating strings ("AAA", "BBB", "CCC") assigned as credit rating outputs in api/ or services/.

Exits with code 1 and prints exact file and line numbers if any regression is detected.
"""

import ast
import os
import sys
from pathlib import Path
from typing import List

# Base directories to audit
AUDIT_DIRS = ["core", "api", "schemas", "services"]

# Outdated Beneish coefficients
OUTDATED_BENEISH_WEIGHTS = {4.037, 0.0327}

# External agency ratings to prevent from being assigned as rating outputs
AGENCY_RATINGS = {"AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CCC/D"}


class ASTAuditor:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.errors: List[str] = []

    def log_error(self, file_path: Path, lineno: int, message: str) -> None:
        try:
            rel_path = file_path.relative_to(self.repo_root)
        except ValueError:
            rel_path = file_path
        self.errors.append(f"{rel_path}:{lineno}: {message}")

    def audit_all(self) -> int:
        for dir_name in AUDIT_DIRS:
            target_dir = self.repo_root / dir_name
            if not target_dir.exists():
                continue
            for py_file in target_dir.rglob("*.py"):
                self.audit_file(py_file)

        if self.errors:
            print("=" * 70, file=sys.stderr)
            print("AST AUDITOR REGRESSION DETECTED:", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            for err in self.errors:
                print(f"  [ERROR] {err}", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            print(f"Total regressions detected: {len(self.errors)}", file=sys.stderr)
            return 1

        print("AST Auditor: All checks passed. 0 architectural regressions found.")
        return 0

    def audit_file(self, file_path: Path) -> None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content, filename=str(file_path))
        except Exception as e:
            self.log_error(file_path, 1, f"Failed to parse AST: {e}")
            return

        rel_path = file_path.relative_to(self.repo_root)
        parts = rel_path.parts

        is_api = "api" in parts
        is_services = "services" in parts
        is_merton = file_path.name == "merton.py" and "core" in parts

        for node in ast.walk(tree):
            # 1. Outdated Beneish weights (4.037 or 0.0327) in any audited dir
            self.check_beneish_weights(file_path, node)

            # 2. Hardcoded 100_000 or 100000 EAD floor clamps in api/
            if is_api:
                self.check_ead_floor_clamp(file_path, node)

            # 3. Hardcoded solver_converged = True or max(v_a, clamps in core/merton.py
            if is_merton:
                self.check_merton_regressions(file_path, node)

            # 4. Generic except Exception returning HTTP status_code=422 in api/
            if is_api:
                self.check_exception_handling(file_path, node)

            # 5. External rating strings assigned as credit rating outputs in api/ or services/
            if is_api or is_services:
                self.check_external_ratings(file_path, node)

    def check_beneish_weights(self, file_path: Path, node: ast.AST) -> None:
        """Check for outdated Beneish coefficients 4.037 or 0.0327."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            for bad_weight in OUTDATED_BENEISH_WEIGHTS:
                if abs(node.value - bad_weight) < 1e-4:
                    self.log_error(
                        file_path,
                        node.lineno,
                        f"Outdated Beneish weight detected: {node.value} (expected 4.679 for TATA or -0.327 for LVGI)",
                    )

    def check_ead_floor_clamp(self, file_path: Path, node: ast.AST) -> None:
        """Check for hardcoded 100_000 EAD floor clamp in api/."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value in (100000, 100000.0, 100_000):
                self.log_error(
                    file_path,
                    node.lineno,
                    f"Hardcoded 100,000 EAD floor clamp detected: {node.value}",
                )

    def check_merton_regressions(self, file_path: Path, node: ast.AST) -> None:
        """Check for hardcoded solver_converged = True or max(v_a, clamps in core/merton.py."""
        # Check variable assignment solver_converged = True
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "solver_converged":
                    if isinstance(node.value, ast.Constant) and node.value.value is True:
                        self.log_error(
                            file_path,
                            node.lineno,
                            "Hardcoded 'solver_converged = True' assignment detected in core/merton.py",
                        )

        # Check max(v_a, ...) clamps
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "max":
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id == "v_a":
                        self.log_error(
                            file_path,
                            node.lineno,
                            "Hardcoded 'max(v_a, ...)' clamp detected in core/merton.py",
                        )

            # Check keyword argument solver_converged=True in MertonOutput calls outside zero_debt_exact
            if isinstance(node.func, ast.Name) and node.func.id == "MertonOutput":
                has_hardcoded_true = False
                is_zero_debt = False
                for kw in node.keywords:
                    if kw.arg == "solver_converged" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        has_hardcoded_true = True
                    if kw.arg == "solver_used" and isinstance(kw.value, ast.Constant) and kw.value.value == "zero_debt_exact":
                        is_zero_debt = True
                if has_hardcoded_true and not is_zero_debt:
                    self.log_error(
                        file_path,
                        node.lineno,
                        "Hardcoded 'solver_converged=True' in MertonOutput detected in core/merton.py",
                    )

    def check_exception_handling(self, file_path: Path, node: ast.AST) -> None:
        """Check for generic except Exception returning HTTP status_code=422 in api/."""
        if isinstance(node, ast.ExceptHandler):
            catches_generic = False
            if node.type is not None:
                if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    catches_generic = True

            if catches_generic:
                for sub_node in ast.walk(node):
                    if isinstance(sub_node, ast.Raise) and isinstance(sub_node.exc, ast.Call):
                        func_name = ""
                        if isinstance(sub_node.exc.func, ast.Name):
                            func_name = sub_node.exc.func.id
                        elif isinstance(sub_node.exc.func, ast.Attribute):
                            func_name = sub_node.exc.func.attr

                        if "HTTPException" in func_name:
                            is_422 = False
                            for kw in sub_node.exc.keywords:
                                if kw.arg == "status_code":
                                    if isinstance(kw.value, ast.Constant) and kw.value.value == 422:
                                        is_422 = True
                                    elif isinstance(kw.value, ast.Attribute) and "422" in kw.value.attr:
                                        is_422 = True
                            if sub_node.exc.args:
                                first_arg = sub_node.exc.args[0]
                                if isinstance(first_arg, ast.Constant) and first_arg.value == 422:
                                    is_422 = True
                                elif isinstance(first_arg, ast.Attribute) and "422" in first_arg.attr:
                                    is_422 = True

                            if is_422:
                                self.log_error(
                                    file_path,
                                    sub_node.lineno,
                                    "Generic 'except Exception' returning HTTP status_code=422 in api/",
                                )

    def check_external_ratings(self, file_path: Path, node: ast.AST) -> None:
        """Check for external rating strings ('AAA', 'BBB', 'CCC', etc.) assigned as credit rating outputs."""
        rating_target_names = {
            "rating",
            "credit_rating",
            "composite_rating",
            "band",
            "internal_risk_band",
        }

        # 1. Variable assignment: rating = "AAA", credit_rating = "BBB", etc.
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                target_id = ""
                if isinstance(target, ast.Name):
                    target_id = target.id
                elif isinstance(target, ast.Attribute):
                    target_id = target.attr

                if any(name in target_id.lower() for name in rating_target_names):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        if node.value.value.upper() in AGENCY_RATINGS:
                            self.log_error(
                                file_path,
                                node.lineno,
                                f"External agency rating '{node.value.value}' assigned to '{target_id}' in credit output",
                            )

        # 2. Dict key assignment: {"credit_rating": "AAA", ...}
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    if any(name in k.value.lower() for name in rating_target_names):
                        if isinstance(v, ast.Constant) and isinstance(v.value, str):
                            if v.value.upper() in AGENCY_RATINGS:
                                self.log_error(
                                    file_path,
                                    k.lineno,
                                    f"External agency rating '{v.value}' in dictionary key '{k.value}'",
                                )

        # 3. Call keyword arguments: ComprehensiveCreditResponse(credit_rating="AAA", ...)
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg and any(name in kw.arg.lower() for name in rating_target_names):
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        if kw.value.value.upper() in AGENCY_RATINGS:
                            self.log_error(
                                file_path,
                                kw.lineno,
                                f"External agency rating '{kw.value.value}' in keyword argument '{kw.arg}'",
                            )


def main():
    repo_root = Path(__file__).resolve().parent.parent
    auditor = ASTAuditor(repo_root)
    sys.exit(auditor.audit_all())


if __name__ == "__main__":
    main()
