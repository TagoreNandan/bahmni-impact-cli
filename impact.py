import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Lightweight change impact & risk analysis for Java codebases.")
console = Console()

RISK_KEYWORDS = [
    # Domain-critical terms (Bahmni/OpenMRS flavored)
    "patient", "identifier", "encounter", "order", "visit", "appointment",
    "provider", "obs", "concept", "drug", "lab",
    # Behavioral/validation signals
    "validate", "validation", "required", "not null", "ensure", "assert",
    # Stability indicators
    "exception", "throw", "transaction", "dao"
]

# Regex patterns for dependency detection
PATTERNS = {
    "import": lambda cls: re.compile(rf"\bimport\s+[\w\.]+\b{re.escape(cls)}\b\s*;"),
    "new": lambda cls: re.compile(rf"\bnew\s+{re.escape(cls)}\s*\("),
    "static_ref": lambda cls: re.compile(rf"\b{re.escape(cls)}\s*\."),
    "type_ref": lambda cls: re.compile(rf"\bextends\s+{re.escape(cls)}\b|\bimplements\s+{re.escape(cls)}\b"),
    "method_ref": lambda cls: re.compile(rf"\b{re.escape(cls)}\s*\("),
}

EXCLUDE_DIRS = {"build", "target", "out", "node_modules", ".git"}


def iter_java_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for file in root.rglob("*.java"):
        parts = set(p.name for p in file.parents)
        if parts & EXCLUDE_DIRS:
            continue
        files.append(file)
    return files


def resolve_target(file_arg: str, project_root: Path) -> Optional[Path]:
    """Resolve the target Java file from provided arg.

    Supports:
    - Absolute or CWD-relative path
    - Path relative to --root
    - Class name (stem) lookup within --root
    """
    # Defensive: empty arg
    if not file_arg or not str(file_arg).strip():
        return None

    # Expand user and vars for any incoming string
    expanded_arg = Path(os.path.expandvars(os.path.expanduser(str(file_arg).strip())))

    # Absolute path: must be a file
    if expanded_arg.is_absolute():
        if expanded_arg.exists() and expanded_arg.is_file():
            return expanded_arg
        return None

    # Path relative to root
    rel = project_root / expanded_arg
    if rel.exists() and rel.is_file():
        return rel

    # Class name lookup by stem
    stem = Path(file_arg).stem
    if not stem:
        return None
    candidates = [f for f in iter_java_files(project_root) if f.stem == stem]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # Prefer standard naming like *Impl first
        impls = [f for f in candidates if f.name.endswith("Impl.java")]
        return impls[0] if impls else candidates[0]

    return None

def find_dependents(root: Path, class_name: str, exclude: Path) -> Dict[str, List[Path]]:
    direct_imports: List[Path] = []
    indirect_refs: List[Path] = []

    # Precompile patterns for speed
    import_re = PATTERNS["import"](class_name)
    other_res = [
        PATTERNS["new"](class_name),
        PATTERNS["static_ref"](class_name),
        PATTERNS["type_ref"](class_name),
        PATTERNS["method_ref"](class_name),
    ]

    for file in iter_java_files(root):
        if file == exclude:
            continue
        try:
            text = file.read_text(errors="ignore")
        except Exception:
            continue

        if import_re.search(text):
            direct_imports.append(file)
            continue

        if any(rx.search(text) for rx in other_res):
            indirect_refs.append(file)

    return {"direct_imports": direct_imports, "indirect_refs": indirect_refs}

def analyze_risk(target: Path, deps: Dict[str, List[Path]]):
    text = target.read_text(errors="ignore").lower()

    # Signals
    keyword_hits = sorted({kw for kw in RISK_KEYWORDS if kw in text})
    throws_count = len(re.findall(r"\bthrow\b", text)) + len(re.findall(r"\bthrows\b", text))
    validate_count = len(re.findall(r"\bvalidate\w*\b", text))
    transactional_hits = len(re.findall(r"@transactional", text))
    null_checks = len(re.findall(r"not null|!=\s*null|==\s*null", text))

    direct = len(deps["direct_imports"]) if deps else 0
    indirect = len(deps["indirect_refs"]) if deps else 0

    # Scoring weights (tuned for explainability over precision)
    score = 0
    score += min(direct * 3, 18)        # strong fan-out
    score += min(indirect * 2, 12)      # weaker references
    score += min(len(keyword_hits) * 2, 12)
    score += min((throws_count + validate_count), 8)
    score += min(transactional_hits * 3, 9)
    score += min(null_checks, 6)

    if score >= 28:
        level = "HIGH"
    elif score >= 16:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "score": score,
        "level": level,
        "keywords": keyword_hits,
        "signals": {
            "throws": throws_count,
            "validate": validate_count,
            "transactional": transactional_hits,
            "null_checks": null_checks,
        },
        "deps": {
            "direct": direct,
            "indirect": indirect,
        },
    }

@app.command()
def impact(
    file: str,
    root: str = ".",
    show_all: bool = typer.Option(False, help="Show full dependent file list"),
    debug: bool = typer.Option(False, help="Print file resolution diagnostics"),
):
    """
    Analyze change impact and risk for a given Java file.
    """
    project_root = Path(os.path.expandvars(os.path.expanduser(root)))
    if not project_root.exists():
        console.print(f"[red]Root not found:[/red] {project_root}")
        raise typer.Exit(2)

    target = resolve_target(file, project_root)
    if not target:
        tips = (
            "[red]Could not resolve target file[/red]\n\n"
            "Tips:\n"
            "• Pass an absolute path\n"
            "• Or path relative to --root\n"
            "• Or class name (e.g., PatientServiceImpl)\n\n"
            f"Current root: {project_root}"
        )
        if debug:
            try:
                java_files = iter_java_files(project_root)
                total = len(java_files)
                stem = Path(file).stem
                exact = [str(f.relative_to(project_root)) for f in java_files if f.stem == stem]
                partial = [str(f.relative_to(project_root)) for f in java_files if stem.lower() in f.name.lower()]
                sample_exact = "\n".join([f"• {p}" for p in exact[:10]]) or "(none)"
                sample_partial = "\n".join([f"• {p}" for p in partial[:10]]) or "(none)"
                tips += (
                    f"\n\nScanned .java files: {total}\n"
                    f"Exact stem matches ({len(exact)}):\n{sample_exact}\n\n"
                    f"Partial name matches ({len(partial)}):\n{sample_partial}"
                )
            except Exception:
                pass

        console.print(Panel.fit(tips, title="File Resolution"))
        raise typer.Exit(1)

    class_name = target.stem
    dependents = find_dependents(project_root, class_name, target)
    risk = analyze_risk(target, dependents)

    console.print(
        Panel.fit(
            f"[bold]Impact Analysis Report[/bold]\n\n"
            f"[cyan]Target:[/cyan] {target.name}\n"
            f"[cyan]Class:[/cyan] {class_name}\n\n"
            f"[yellow]Risk Level:[/yellow] [bold]{risk['level']}[/bold]\n"
            f"[yellow]Risk Score:[/yellow] {risk['score']}",
            title="⚠️ Change Risk"
        )
    )

    table = Table(title="Downstream Impact & Signals")
    table.add_column("Type")
    table.add_column("Count")

    table.add_row("Direct imports", str(risk["deps"]["direct"]))
    table.add_row("Indirect references", str(risk["deps"]["indirect"]))
    table.add_row("Throws/Validate", str(risk["signals"]["throws"] + risk["signals"]["validate"]))
    table.add_row("Transactional annotations", str(risk["signals"]["transactional"]))
    table.add_row("Null checks", str(risk["signals"]["null_checks"]))
    table.add_row("Risk keywords", ", ".join(risk["keywords"]) or "None")

    console.print(table)

    all_dependents: List[Path] = dependents["direct_imports"] + dependents["indirect_refs"]
    if all_dependents:
        console.print("\n[bold]Affected Files:[/bold]")
        to_show = all_dependents if show_all else all_dependents[:15]
        for dep in to_show:
            console.print(f"• {dep}")
        if not show_all and len(all_dependents) > len(to_show):
            console.print(f"\n[dim]{len(all_dependents) - len(to_show)} more... use --show-all to display all[/dim]")

if __name__ == "__main__":
    app()
