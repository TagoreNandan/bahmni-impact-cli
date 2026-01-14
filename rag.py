#!/usr/bin/env python3

"""
Minimal RAG CLI for legacy code intelligence (Bahmni demo)

Design goals:
- Keep it tiny and demo-first (2–3 files only)
- Use local, deterministic embeddings (HashingVectorizer) to avoid heavy models
- Attach risk/impact metadata from the existing impact analyzer (impact.py)
- Output retrieved code + risk context for safer optimization decisions

This is not a chatbot. It's a retrieval aid you can wire into future tooling.
"""

import json
import re
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text
from rich.syntax import Syntax
from sklearn.feature_extraction.text import HashingVectorizer

# Reuse the existing impact analysis to get risk metadata
from impact import iter_java_files, resolve_target, find_dependents, analyze_risk

app = typer.Typer(help="RAG CLI for Bahmni legacy code intelligence (demo)")
console = Console()


# ---------- Simple storage layout ----------
DEFAULT_INDEX_DIR = ".rag_index"
VECTORS_FILE = "vectors.npy"      # shape: (n_chunks, n_features) float32
METADATA_FILE = "index.json"       # list of chunk metadata dicts

# Fixed HashingVectorizer config so we don't need to persist a model
def make_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=16384,          # small, fast, adequate for 2–3 files
        alternate_sign=False,      # keep non-negative for cosine clarity
        analyzer="word",
        ngram_range=(1, 2),        # unigrams + bigrams give better recall
        norm="l2",
        lowercase=True,
    )


@dataclass
class Chunk:
    id: int
    file: str
    start_line: int
    end_line: int
    snippet: str
    risk_level: str
    risk_score: int
    deps: Dict[str, int]  # {direct, indirect}
    keywords: List[str] = field(default_factory=list)
    signals: Dict[str, int] = field(default_factory=dict)  # {throws, validate, transactional, null_checks}


def chunk_text(text: str, lines_per_chunk: int = 30, overlap: int = 8) -> List[Tuple[int, int, str]]:
    """Return list of (start_line, end_line, snippet). Lines are 1-based.

    Simple line windowing is explainable and good enough for this demo.
    """
    lines = text.splitlines()
    n = len(lines)
    chunks: List[Tuple[int, int, str]] = []
    i = 0
    while i < n:
        start = i
        end = min(i + lines_per_chunk, n)
        snippet = "\n".join(lines[start:end])
        # convert to 1-based line numbers for display
        chunks.append((start + 1, end, snippet))
        if end == n:
            break
        i = end - overlap
        if i <= start:  # safety
            i = end
    return chunks


def ensure_index_dir(index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)


def save_index(index_dir: Path, vectors: np.ndarray, chunks: List[Chunk]) -> None:
    ensure_index_dir(index_dir)
    np.save(index_dir / VECTORS_FILE, vectors.astype(np.float32))
    with (index_dir / METADATA_FILE).open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)


def load_index(index_dir: Path) -> Tuple[np.ndarray, List[Chunk]]:
    vectors = np.load(index_dir / VECTORS_FILE)
    with (index_dir / METADATA_FILE).open("r", encoding="utf-8") as f:
        raw = json.load(f)
        chunks = [Chunk(**item) for item in raw]
    return vectors, chunks


def cosine_top_k(matrix: np.ndarray, query_vec: np.ndarray, k: int = 3) -> List[int]:
    """Return indices of top-k most similar rows via dot product.

    Assumes rows and query are L2-normalized (HashingVectorizer with norm="l2").
    """
    sims = matrix @ query_vec
    if k >= len(sims):
        return list(np.argsort(-sims))
    return list(np.argpartition(-sims, k)[:k][np.argsort(-sims[np.argpartition(-sims, k)[:k]])])


@app.command()
def ingest(
    files: List[str] = typer.Argument(..., help="2–3 Java files to index"),
    root: str = typer.Option(".", help="Project root for dependency scanning"),
    index_dir: str = typer.Option(DEFAULT_INDEX_DIR, help="Where to store the vector index"),
    chunk_lines: int = typer.Option(30, help="Lines per chunk"),
    overlap: int = typer.Option(8, help="Overlapping lines between chunks"),
):
    """Ingest Java files: chunk, embed, and attach risk metadata.

    Example:
      python3 rag.py ingest path/PatientService.java path/PatientServiceImpl.java --root /path/to/bahmni-core
    """
    project_root = Path(os.path.expanduser(os.path.expandvars(root)))
    index_path = Path(index_dir)
    vectorizer = make_vectorizer()

    all_chunks: List[Chunk] = []
    texts: List[str] = []

    for file in files:
        target = resolve_target(file, project_root)
        if not target:
            console.print(f"[red]Could not resolve:[/red] {file}")
            raise typer.Exit(2)

        # Risk/impact per-file using existing analyzer
        deps = find_dependents(project_root, target.stem, target)
        risk = analyze_risk(target, deps)

        text = target.read_text(errors="ignore")
        for (start, end, snippet) in chunk_text(text, lines_per_chunk=chunk_lines, overlap=overlap):
            chunk = Chunk(
                id=len(all_chunks),
                file=str(target),
                start_line=start,
                end_line=end,
                snippet=snippet,
                risk_level=risk["level"],
                risk_score=int(risk["score"]),
                deps={"direct": int(risk["deps"]["direct"]), "indirect": int(risk["deps"]["indirect"])},
                keywords=list(risk.get("keywords", [])),
                signals=dict(risk.get("signals", {})),
            )
            all_chunks.append(chunk)
            texts.append(snippet)

    # Build embedding matrix (L2-normalized by vectorizer)
    X = vectorizer.transform(texts)
    vectors = X.toarray().astype(np.float32)

    save_index(index_path, vectors, all_chunks)

    console.print(
        Panel.fit(
            f"Indexed [bold]{len(files)}[/bold] files into [bold]{len(all_chunks)}[/bold] chunks\n"
            f"Index dir: [green]{index_path}[/green]",
            title="RAG Ingest Complete"
        )
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about the indexed code"),
    index_dir: str = typer.Option(DEFAULT_INDEX_DIR, help="Where the vector index is stored"),
    k: int = typer.Option(3, help="How many results to show"),
    simple: bool = typer.Option(False, help="Print concise text output instead of panels"),
):
    """Retrieve relevant code and risk context to answer focused questions.

    Example:
      python3 rag.py ask "Why is PatientService risky?"
    """
    index_path = Path(index_dir)
    if not (index_path / VECTORS_FILE).exists() or not (index_path / METADATA_FILE).exists():
        console.print(f"[red]No index found in[/red] {index_path}. Run: python3 rag.py ingest ...")
        raise typer.Exit(2)

    vectors, chunks = load_index(index_path)
    vectorizer = make_vectorizer()
    qv = vectorizer.transform([question]).toarray().astype(np.float32)[0]

    top_idx = cosine_top_k(vectors, qv, k=k)

    for rank, idx in enumerate(top_idx, start=1):
        c = chunks[idx]
        header = f"[{rank}] {Path(c.file).name}  •  Risk: {c.risk_level} (score {c.risk_score})  •  Deps: d={c.deps['direct']}, i={c.deps['indirect']}"

        # Grounded explanation: based on retrieved snippet + risk metadata
        explanation = (
            "Retrieved by semantic similarity. Risk context reflects dependency fan-out and defensive signals. "
            "Preserve documented business rules and security/transaction semantics when changing this code."
        )

        # Heuristic extraction of business rules & signals from the snippet
        def _strip_tags_and_comments(s: str) -> str:
            s = re.sub(r"<[^>]+>", "", s)  # strip HTML tags
            s = re.sub(r"^\s*(/\*+|\*+/?|//)\s*", "", s)  # comment markers
            s = re.sub(r"\s+", " ", s)
            return s.strip()

        rules: List[str] = []
        seen_norm: set = set()
        lines = c.snippet.splitlines()

        # 1) Collect Javadoc sentences with clear guidance (Should/must/required)
        in_javadoc = False
        javadoc_buf: List[str] = []
        def _flush_javadoc():
            text = _strip_tags_and_comments(" ".join(javadoc_buf))
            if not text:
                return
            # split on sentence boundaries
            for sent in re.split(r"(?<=[\.!?])\s+", text):
                s = sent.strip()
                if not s:
                    continue
                if s.lower().startswith(("@param", "@return", "@since")):
                    continue
                if any(key in s.lower() for key in ("should", "must", "required")):
                    n = s.lower()
                    if n not in seen_norm:
                        seen_norm.add(n)
                        rules.append(s)
        for ln in lines:
            if "/**" in ln:
                in_javadoc = True
                javadoc_buf = []
            if in_javadoc:
                javadoc_buf.append(ln)
            if in_javadoc and "*/" in ln:
                in_javadoc = False
                _flush_javadoc()

        # 2) Annotations: @Authorized, @Transactional
        auth_re = re.compile(r"@Authorized\((.*?)\)")
        tran_re = re.compile(r"@Transactional\((.*?)\)")
        for ln in lines:
            m = auth_re.search(ln)
            if m:
                raw = m.group(1)
                privs = re.findall(r"PrivilegeConstants\.(\w+)|\"([^\"]+)\"", raw)
                vals = [a or b for (a, b) in privs]
                if vals:
                    s = f"Requires privilege: {', '.join(vals)}"
                else:
                    s = "Requires privilege (see @Authorized)"
                n = s.lower()
                if n not in seen_norm:
                    seen_norm.add(n)
                    rules.append(s)
            m2 = tran_re.search(ln)
            if m2:
                raw = m2.group(1)
                conf = _strip_tags_and_comments(raw)
                s = f"Transactional: {conf}" if conf else "Transactional"
                n = s.lower()
                if n not in seen_norm:
                    seen_norm.add(n)
                    rules.append(s)

        # 3) Method signature throws
        sig_re = re.compile(r"^(public|protected|private)\b[^{;]*?\)\s*(throws\s+([A-Za-z0-9_,\s]+))?", re.MULTILINE)
        for ln in lines:
            m = sig_re.search(ln)
            if m and m.group(2):
                types = [t.strip() for t in m.group(3).split(',') if t.strip()]
                if types:
                    s = f"Throws: {', '.join(types)}"
                    n = s.lower()
                    if n not in seen_norm:
                        seen_norm.add(n)
                        rules.append(s)

        # 4) Null-check preconditions leading to throws
        for i, ln in enumerate(lines):
            null_m = re.search(r"if\s*\(\s*(\w+)\s*==\s*null\s*\)\s*\{?", ln)
            if null_m:
                var = null_m.group(1)
                window = "\n".join(lines[i:i+5])
                if re.search(r"throw\s+new\s+[A-Za-z0-9_]+\(", window):
                    s = f"Requires: {var} != null"
                    n = s.lower()
                    if n not in seen_norm:
                        seen_norm.add(n)
                        rules.append(s)

        # Finalize: cap size and length; keep most informative first
        cleaned: List[str] = []
        for s in rules:
            s2 = s.strip()
            if len(s2) > 160:
                s2 = s2[:157] + "..."
            cleaned.append(s2)
        highlights = cleaned[:6]

        if simple:
            console.print(f"- File: {Path(c.file).name} (risk {c.risk_level} {c.risk_score}, deps d={c.deps['direct']}, i={c.deps['indirect']})")
            console.print(f"- Lines: {c.start_line}-{c.end_line}")
            if c.keywords:
                console.print(f"- Keywords: {', '.join(sorted(set(c.keywords))[:8])}")
            if highlights:
                console.print("- Rules:")
                for h in highlights[:4]:
                    console.print(f"  - {h}")
            else:
                console.print("- Rules: (none detected)")
            console.print("- Why: matches your query; preserve rules and dependencies.")
        else:
            # Organized metadata block
            meta = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0,1))
            meta.add_column("Field", style="bold cyan", no_wrap=True)
            meta.add_column("Value", overflow="fold")
            meta.add_row("File", str(c.file))
            meta.add_row("Lines", f"{c.start_line}–{c.end_line}")
            meta.add_row("Risk", f"{c.risk_level} (score {c.risk_score})")
            meta.add_row("Deps", f"direct={c.deps['direct']}, indirect={c.deps['indirect']}")
            if c.keywords:
                kw = sorted(set(c.keywords))
                kw_line = " · ".join(kw[:12]) + (" · …" if len(kw) > 12 else "")
                meta.add_row("Keywords", kw_line)

            short_title = f"{rank}. {Path(c.file).name}"
            console.print(Panel(meta, title=short_title))

            # Highlight lines in snippet matching query tokens or key triggers
            tokens = [t.lower() for t in re.findall(r"\w+", question) if len(t) >= 5]
            trigger_words = [
                "@authorized", "@transactional", "throws", "not null", "cannot be null",
                "validate", "assert", "error", "should ", "must ",
            ]
            rel_highlights = set()
            snippet_lines = c.snippet.splitlines()
            for i, ln in enumerate(snippet_lines):
                ll = ln.lower()
                if any(w in ll for w in trigger_words) or any(tok in ll for tok in tokens):
                    rel_highlights.add(i + 1)  # 1-based within snippet

            # Crop snippet around the first highlighted line for tighter context
            if rel_highlights:
                anchor = sorted(rel_highlights)[0] - 1  # zero-based within snippet
            else:
                anchor = max(len(snippet_lines) // 2, 0)
            before, after = 6, 6
            start_off = max(anchor - before, 0)
            end_off = min(anchor + after + 1, len(snippet_lines))
            cropped = snippet_lines[start_off:end_off]
            new_start_line = c.start_line + start_off

            # Recompute file-based highlight lines for the cropped range
            file_highlights = set()
            for rel in rel_highlights:
                idx0 = rel - 1
                if start_off <= idx0 < end_off:
                    file_highlights.add(c.start_line + idx0)

            # Best-effort method name detection within cropped lines (lookback inside crop)
            method_name = None
            method_pat = re.compile(r"\b(public|protected|private)\b[^{;]*?\b(\w+)\s*\(.*\)")
            for ln in reversed(cropped[: (anchor - start_off + 1)]):
                m = method_pat.search(ln)
                if m:
                    method_name = m.group(2)
                    break
            if method_name:
                table.add_row("Method", method_name)

            syntax = Syntax(
                "\n".join(cropped),
                "java",
                line_numbers=True,
                start_line=new_start_line,
                word_wrap=False,
                highlight_lines=sorted(file_highlights) if file_highlights else None,
            )
            console.print(Panel(syntax, title="Relevant Code Snippet"))

            # Critical Business Rules (declarative)
            def _to_declarative(s: str) -> str:
                s2 = re.sub(r"(?i)\bshould\b", "Must", s)
                s2 = re.sub(r"(?i)\bshould\s+be\b", "Must be", s2)
                s2 = re.sub(r"(?i)\bshould\s+return\b", "Must return", s2)
                s2 = re.sub(r"(?i)is null", "must not be null", s2)
                s2 = re.sub(r"\s+", " ", s2).strip()
                return s2

            rules_decl: List[str] = []
            if highlights:
                rules_decl = [_to_declarative(h) for h in highlights]
                rules_panel = Panel.fit("\n".join(f"- {h}" for h in rules_decl), title="Critical Business Rules to Preserve")
                console.print(rules_panel)

            # Why This Is Risky
            risky_bullets: List[str] = []
            risky_bullets.append(f"Dependency fan-out: direct={c.deps['direct']}, indirect={c.deps['indirect']}")
            if c.signals:
                tv = (c.signals.get("throws", 0) + c.signals.get("validate", 0))
                if tv:
                    risky_bullets.append(f"Validation/throws present: {tv}")
                tc = c.signals.get("transactional", 0)
                if tc:
                    risky_bullets.append(f"Transactional annotations: {tc}")
                nc = c.signals.get("null_checks", 0)
                if nc:
                    risky_bullets.append(f"Null-checks: {nc}")
            if c.keywords:
                dom = [w for w in c.keywords if w in ("patient", "encounter", "identifier", "visit", "provider", "order", "obs")]
                if dom:
                    risky_bullets.append(f"Domain-critical terms: {', '.join(sorted(set(dom))[:5])}")
            console.print(Panel.fit("\n".join(f"- {b}" for b in risky_bullets), title="Why This Is Risky"))

            # Optimization Guidance (Non-Code)
            guidance: List[str] = []
            # Safe areas
            if any("log." in ln.lower() for ln in snippet_lines):
                guidance.append("Logging messages/levels can be adjusted; do not remove validations.")
            guidance.append("Micro-optimizations only (locals/allocations) without behavior change.")
            # Explicit warnings based on signals
            if c.signals.get("transactional", 0) > 0:
                guidance.append("Do NOT alter transaction boundaries or readOnly flags.")
            if (c.signals.get("throws", 0) + c.signals.get("validate", 0)) > 0:
                guidance.append("Do NOT bypass validations or exception paths.")
            # If any privilege rule detected
            if any("Requires privilege" in r for r in rules_decl):
                guidance.append("Do NOT relax authorization checks.")
            console.print(Panel.fit("\n".join(f"- {g}" for g in guidance), title="Optimization Guidance (Safe)"))

            console.print(Panel.fit(explanation, title="Brief Explanation"))


@app.command()
def scan(
    query: str = typer.Argument(..., help="Substring to search in Java file names (e.g., PatientService)"),
    root: str = typer.Option(".", help="Project root to scan for .java files"),
    limit: int = typer.Option(30, help="Max results to show"),
):
    """List Java files under --root whose file name contains the query.

    Use this to discover exact paths or stems before calling `ingest`.
    """
    project_root = Path(os.path.expanduser(os.path.expandvars(root)))
    if not project_root.exists():
        console.print(f"[red]Root not found:[/red] {project_root}")
        raise typer.Exit(2)

    files = iter_java_files(project_root)
    q = query.lower()
    matches = [f for f in files if q in f.name.lower()]
    matches = sorted(matches, key=lambda p: p.name.lower())

    if not matches:
        console.print(f"No matches for '{query}' under {project_root}")
        raise typer.Exit(1)

    shown = matches[:limit]
    table = Table(title=f"Matches for '{query}' under {project_root}")
    table.add_column("Stem")
    table.add_column("Relative Path")
    for p in shown:
        rel = str(p.relative_to(project_root))
        table.add_row(p.stem, rel)

    console.print(table)
    if len(matches) > len(shown):
        console.print(f"[dim]{len(matches) - len(shown)} more... use --limit to show more[/dim]")


if __name__ == "__main__":
    app()
