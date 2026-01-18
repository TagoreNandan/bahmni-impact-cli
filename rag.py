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


# ---------- Shared formatting helpers ----------
def _extract_rules(snippet: str) -> List[str]:
    def _strip_tags_and_comments(s: str) -> str:
        s = re.sub(r"<[^>]+>", "", s)  # strip HTML tags
        s = re.sub(r"^\s*(/\*+|\*+/?|//)\s*", "", s)  # comment markers
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    rules: List[str] = []
    seen_norm: set = set()
    lines = snippet.splitlines()

    # 1) Collect Javadoc sentences with clear guidance (Should/must/required)
    in_javadoc = False
    javadoc_buf: List[str] = []

    def _flush_javadoc():
        text = _strip_tags_and_comments(" ".join(javadoc_buf))
        if not text:
            return
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
    tran_re = re.compile(r"@Transactional(?:\((.*?)\))?")
    for ln in lines:
        m = auth_re.search(ln)
        if m:
            raw = m.group(1)
            privs = re.findall(r"PrivilegeConstants\.(\w+)|\"([^\"]+)\"", raw)
            vals = [a or b for (a, b) in privs]
            s = f"Requires privilege: {', '.join(vals)}" if vals else "Requires privilege (see @Authorized)"
            n = s.lower()
            if n not in seen_norm:
                seen_norm.add(n)
                rules.append(s)
        m2 = tran_re.search(ln)
        if m2:
            raw = m2.group(1) or ""
            conf = _strip_tags_and_comments(raw)
            s = f"Transactional: {conf}" if conf else "Transactional"
            n = s.lower()
            if n not in seen_norm:
                seen_norm.add(n)
                rules.append(s)

    # 3) Method signature throws
    sig_re = re.compile(r"^(public|protected|private)\b[^\{;]*?\)\s*(throws\s+([A-Za-z0-9_,\s]+))?", re.MULTILINE)
    for ln in lines:
        m = sig_re.search(ln)
        if m and m.group(2):
            types = [t.strip() for t in (m.group(3) or '').split(',') if t.strip()]
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
            window = "\n".join(lines[i:i + 5])
            if re.search(r"throw\s+new\s+[A-Za-z0-9_]+\(", window):
                s = f"Requires: {var} != null"
                n = s.lower()
                if n not in seen_norm:
                    seen_norm.add(n)
                    rules.append(s)

    cleaned: List[str] = []
    for s in rules:
        s2 = s.strip()
        if len(s2) > 160:
            s2 = s2[:157] + "..."
        cleaned.append(s2)
    return cleaned[:6]


def _rules_to_invariants(rules: List[str]) -> List[str]:
    def _to_declarative(s: str) -> str:
        s2 = re.sub(r"(?i)\bshould\b", "Must", s)
        s2 = re.sub(r"(?i)\bshould\s+be\b", "Must be", s2)
        s2 = re.sub(r"(?i)\bshould\s+return\b", "Must return", s2)
        s2 = re.sub(r"(?i)is null", "must not be null", s2)
        s2 = re.sub(r"\s+", " ", s2).strip()
        return s2

    out: List[str] = []
    for s in rules:
        m = re.match(r"Requires privilege:\s*(.+)", s)
        if m:
            out.append(f"Only users with {m.group(1)} may execute.")
            continue
        m = re.match(r"Transactional:\s*(.*)", s)
        if m:
            conf = (m.group(1) or '').strip()
            out.append(f"Executes within a transaction; preserve configuration ({conf})." if conf else "Executes within a transaction; do not change boundaries.")
            continue
        m = re.match(r"Throws:\s*(.+)", s)
        if m:
            out.append(f"Declared to throw {m.group(1)}; do not remove exception paths.")
            continue
        m = re.match(r"Requires:\s*(\w+)\s*!=\s*null", s)
        if m:
            out.append(f"Input {m.group(1)} must not be null.")
            continue
        out.append(_to_declarative(s))
    return out


def _build_risk_bullets(c: Chunk) -> List[str]:
    bullets: List[str] = []
    bullets.append(f"Dependency fan-out (direct={c.deps['direct']}, indirect={c.deps['indirect']}): altering behavior propagates to callers.")
    tv = (c.signals.get("throws", 0) + c.signals.get("validate", 0))
    if tv:
        bullets.append("Validations/exceptions present: changing logic can break established exception contracts.")
    if c.signals.get("transactional", 0):
        bullets.append("Transactional annotations present: data consistency depends on transaction boundaries.")
    if c.signals.get("null_checks", 0):
        bullets.append("Null-checks present: non-null invariants must be preserved.")
    if c.keywords:
        dom = [w for w in c.keywords if w in ("patient", "encounter", "identifier", "visit", "provider", "order", "obs")]
        if dom:
            bullets.append(f"Involves core domain entities: {', '.join(sorted(set(dom))[:5])}.")
    return bullets


def _build_guidance(c: Chunk, snippet: str, rules_decl: List[str]) -> Tuple[List[str], List[str]]:
    safe: List[str] = []
    do_not: List[str] = []
    if any("log." in ln.lower() for ln in snippet.splitlines()):
        safe.append("Adjust logging messages/levels.")
    safe.append("Consider micro-optimizations (locals/allocations) only; no behavior change.")
    if c.signals.get("transactional", 0) > 0:
        do_not.append("Transaction boundaries or readOnly flags.")
    if (c.signals.get("throws", 0) + c.signals.get("validate", 0)) > 0:
        do_not.append("Validation logic and exception paths.")
    if any("Only users with" in r for r in rules_decl):
        do_not.append("Authorization checks and privileges.")
    return safe, do_not


def _brief_explanation(c: Chunk, start_line: int, end_line: int) -> str:
    parts: List[str] = []
    if c.signals.get("transactional", 0):
        parts.append("transactional boundaries")
    tv = (c.signals.get("throws", 0) + c.signals.get("validate", 0))
    if tv:
        parts.append("validation/exception safeguards")
    if c.signals.get("null_checks", 0):
        parts.append("null-check invariants")
    if not parts:
        parts.append("existing safeguards")
    core = " and ".join(parts[:2])
    return (
        f"{Path(c.file).name} [{start_line}–{end_line}] includes {core} with dependency fan-out. "
        f"Changing behavior risks breaking contracts and downstream callers; prefer non-functional optimizations only."
    )


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
    best: bool = typer.Option(False, help="Return only the single best result (alias of --k 1)"),
    unique_per_file: bool = typer.Option(False, help="At most one top result per file"),
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

    # Compute similarities explicitly so we can filter/dedupe before slicing
    sims = vectors @ qv
    ordered = list(np.argsort(-sims))

    # Dedupe by file if requested: keep the highest-similarity chunk per file
    if unique_per_file:
        seen = set()
        deduped = []
        for idx in ordered:
            f = str(Path(chunks[idx].file))
            if f in seen:
                continue
            seen.add(f)
            deduped.append(idx)
        ordered = deduped

    # If --best is set, override k to 1
    limit = 1 if best else max(1, int(k))
    top_idx = ordered[:limit]

    for rank, idx in enumerate(top_idx, start=1):
        c = chunks[idx]
        # Verdict derived from risk level (formatting only)
        verdict_map = {
            "HIGH": "BEHAVIORAL OPTIMIZATION IS NOT SAFE",
            "MEDIUM": "CAUTION: AVOID BEHAVIORAL CHANGES",
            "LOW": "SAFE FOR NON-FUNCTIONAL OPTIMIZATIONS ONLY",
        }
        verdict = verdict_map.get(c.risk_level, "REVIEW SAFEGUARDS BEFORE CHANGES")
        header = f"[bold]Verdict:[/bold] [red]{verdict}[/red]\n[{rank}] {Path(c.file).name}  •  Risk: {c.risk_level} (score {c.risk_score})  •  Deps: d={c.deps['direct']}, i={c.deps['indirect']}"

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
        tran_re = re.compile(r"@Transactional(?:\((.*?)\))?")
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
                raw = m2.group(1) or ""
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
            method_pat = re.compile(r"\b(public|protected|private)\b[^\{;]*?\b(\w+)\s*\(.*\)")
            for ln in reversed(cropped[: (anchor - start_off + 1)]):
                m = method_pat.search(ln)
                if m:
                    method_name = m.group(2)
                    break

            # Organized metadata block (after we know the cropped range and method)
            meta = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0,1))
            meta.add_column("Field", style="bold cyan", no_wrap=True)
            meta.add_column("Value", overflow="fold")
            meta.add_row("File", str(c.file))
            meta.add_row("Lines", f"{new_start_line}–{new_start_line + len(cropped) - 1}")
            if method_name:
                meta.add_row("Method", method_name)
            meta.add_row("Risk", f"{c.risk_level} (score {c.risk_score})")
            meta.add_row("Deps", f"direct={c.deps['direct']}, indirect={c.deps['indirect']}")
            if c.keywords:
                kw = sorted(set(c.keywords))
                kw_line = " · ".join(kw[:12]) + (" · …" if len(kw) > 12 else "")
                meta.add_row("Keywords", kw_line)

            short_title = f"{rank}. {Path(c.file).name}"
            console.print(Panel(meta, title=short_title))

            syntax = Syntax(
                "\n".join(cropped),
                "java",
                line_numbers=True,
                start_line=new_start_line,
                word_wrap=False,
                highlight_lines=sorted(file_highlights) if file_highlights else None,
            )
            console.print(Panel(syntax, title="Relevant Code Snippet"))

            # Section separator and header: Critical Business Rules
            console.rule("Critical Business Rules to Preserve")
            # Critical Business Rules (declarative) — grounded in retrieved text
            def _to_declarative(s: str) -> str:
                s2 = re.sub(r"(?i)\bshould\b", "Must", s)
                s2 = re.sub(r"(?i)\bshould\s+be\b", "Must be", s2)
                s2 = re.sub(r"(?i)\bshould\s+return\b", "Must return", s2)
                s2 = re.sub(r"(?i)is null", "must not be null", s2)
                s2 = re.sub(r"\s+", " ", s2).strip()
                return s2

            def _to_invariant(s: str) -> str:
                m = re.match(r"Requires privilege:\s*(.+)", s)
                if m:
                    return f"Only users with {m.group(1)} may execute."
                m = re.match(r"Transactional:\s*(.*)", s)
                if m:
                    conf = m.group(1).strip()
                    if conf:
                        return f"Executes within a transaction; preserve configuration ({conf})."
                    return "Executes within a transaction; do not change boundaries."
                m = re.match(r"Throws:\s*(.+)", s)
                if m:
                    return f"Declared to throw {m.group(1)}; do not remove exception paths."
                m = re.match(r"Requires:\s*(\w+)\s*!=\s*null", s)
                if m:
                    return f"Input {m.group(1)} must not be null."
                return _to_declarative(s)

            rules_decl: List[str] = []
            if highlights:
                rules_decl = [_to_invariant(h) for h in highlights]
                rules_panel = Panel.fit("\n".join(f"- {h}" for h in rules_decl))
                console.print(rules_panel)

            # Why This Is Risky
            risky_bullets: List[str] = []
            risky_bullets.append(f"Dependency fan-out (direct={c.deps['direct']}, indirect={c.deps['indirect']}): altering behavior propagates to callers.")
            tv = (c.signals.get("throws", 0) + c.signals.get("validate", 0))
            if tv:
                risky_bullets.append(f"Validations/exceptions ({tv}): changing logic can break established exception contracts.")
            tc = c.signals.get("transactional", 0)
            if tc:
                risky_bullets.append("Transactional annotations present: data consistency depends on transaction boundaries.")
            nc = c.signals.get("null_checks", 0)
            if nc:
                risky_bullets.append("Null-checks present: non-null invariants must be preserved.")
            if c.keywords:
                dom = [w for w in c.keywords if w in ("patient", "encounter", "identifier", "visit", "provider", "order", "obs")]
                if dom:
                    risky_bullets.append(f"Involves core domain entities: {', '.join(sorted(set(dom))[:5])}.")
            # Section separator and header: Why This Is Risky
            console.rule("Why This Is Risky")
            console.print(Panel.fit("\n".join(f"- {b}" for b in risky_bullets)))

            # Optimization Guidance (Non-Code) — conservative and grounded in metadata/signals
            safe_guidance: List[str] = []
            if any("log." in ln.lower() for ln in snippet_lines):
                safe_guidance.append("Adjust logging messages/levels.")
            safe_guidance.append("Consider micro-optimizations (locals/allocations) only; no behavior change.")

            do_not_touch: List[str] = []
            if c.signals.get("transactional", 0) > 0:
                do_not_touch.append("Transaction boundaries or readOnly flags.")
            if (c.signals.get("throws", 0) + c.signals.get("validate", 0)) > 0:
                do_not_touch.append("Validation logic and exception paths.")
            if any("Only users with" in r for r in rules_decl):
                do_not_touch.append("Authorization checks and privileges.")

            # Section separator and header: Optimization Guidance
            console.rule("Optimization Guidance")
            guidance_lines: List[str] = []
            if safe_guidance:
                guidance_lines.append("[green]Safe:[/green]")
                guidance_lines += [f"- {g}" for g in safe_guidance]
            if do_not_touch:
                guidance_lines.append("[bold red]DO NOT TOUCH:[/bold red]")
                guidance_lines += [f"‼ {g}" for g in do_not_touch]
            console.print(Panel.fit("\n".join(guidance_lines)))

            # Brief, engineering-style summary tied to file + lines
            brief_parts: List[str] = []
            if tc:
                brief_parts.append("transactional boundaries")
            if tv:
                brief_parts.append("validation/exception safeguards")
            if nc:
                brief_parts.append("null-check invariants")
            if not brief_parts:
                brief_parts.append("existing safeguards")
            brief_core = " and ".join(brief_parts[:2])
            line_range = f"{new_start_line}–{new_start_line + len(cropped) - 1}"
            brief = (
                f"{Path(c.file).name} [{line_range}] includes {brief_core} with dependency fan-out. "
                f"Changing behavior risks breaking contracts and downstream callers; prefer non-functional optimizations only."
            )
            # Section separator and header: Brief Explanation
            console.rule("Brief Explanation")
            console.print(Panel.fit(brief))
            # Minimal, neutral guidance footer
            console.print("[dim]Next step: restrict changes to non-functional optimizations only.[/dim]")


@app.command()
def chat(
    index_dir: str = typer.Option(DEFAULT_INDEX_DIR, help="Where the vector index is stored"),
    unique_per_file: bool = typer.Option(False, help="At most one top result per file"),
):
    """Interactive, stateful RAG session for legacy-system reasoning.

    Requirements:
    - Every turn triggers retrieval (no free-form answers)
    - Session context biases retrieval toward current focus file(s)
    - No long-term memory; state lives only during this process
    """
    index_path = Path(index_dir)
    if not (index_path / VECTORS_FILE).exists() or not (index_path / METADATA_FILE).exists():
        console.print(f"[red]No index found in[/red] {index_path}. Run: python3 rag.py ingest ...")
        raise typer.Exit(2)

    vectors, chunks = load_index(index_path)
    vectorizer = make_vectorizer()
    focus_files: List[str] = []

    console.print(Panel.fit("Interactive RAG session. Type your question. Type 'exit' to quit.", title="rag chat"))
    while True:
        try:
            q = input("› ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nExiting chat.")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        qv = vectorizer.transform([q]).toarray().astype(np.float32)[0]
        sims = vectors @ qv

        # Bias scores toward current focus files (if any), but still retrieve globally
        if focus_files:
            focus_set = set(focus_files)
            for i, c in enumerate(chunks):
                if Path(c.file) .__str__() in focus_set or Path(c.file).name in focus_set:
                    sims[i] *= 1.15  # modest bias

        ordered = list(np.argsort(-sims))
        if unique_per_file:
            seen = set()
            deduped = []
            for idx in ordered:
                f = str(Path(chunks[idx].file))
                if f in seen:
                    continue
                seen.add(f)
                deduped.append(idx)
            ordered = deduped

        if not ordered:
            console.print("No results. Try a different question or re-ingest.")
            continue

        idx = ordered[0]
        c = chunks[idx]

        # Update focus to this top file (by absolute path and name for robustness)
        focus_files = [str(Path(c.file)), Path(c.file).name]

        # Derive structured sections
        rules_raw = _extract_rules(c.snippet)
        rules_decl = _rules_to_invariants(rules_raw)
        risky = _build_risk_bullets(c)
        safe, do_not = _build_guidance(c, c.snippet, rules_decl)
        line_range = f"{c.start_line}–{c.end_line}"
        brief = _brief_explanation(c, c.start_line, c.end_line)

        # Verdict derived from risk level (formatting only)
        verdict_map = {
            "HIGH": "BEHAVIORAL OPTIMIZATION IS NOT SAFE",
            "MEDIUM": "CAUTION: AVOID BEHAVIORAL CHANGES",
            "LOW": "SAFE FOR NON-FUNCTIONAL OPTIMIZATIONS ONLY",
        }
        verdict = verdict_map.get(c.risk_level, "Review safeguards before changes")

        # Render stable text with separators and emphasis
        sep = "\u2500" * 70  # heavy horizontal line
        lines: List[str] = []
        lines.append(f"[bold]Verdict:[/bold] [red]{verdict}[/red]")
        lines.append(f"[bold]File:[/bold] {Path(c.file).name}")
        lines.append(f"[bold]Lines:[/bold] {line_range}")
        lines.append(f"[bold]Risk:[/bold] [red]{c.risk_level}[/red] (score {c.risk_score})")
        lines.append(f"[bold]Dependencies:[/bold] direct={c.deps['direct']}, indirect={c.deps['indirect']}")
        lines.append("")

        # Critical Business Rules
        lines.append(sep)
        lines.append("[bold]Critical Business Rules to Preserve[/bold]")
        lines.append(sep)
        if rules_decl:
            lines += [f"- {r}" for r in rules_decl]
        else:
            lines.append("- (none detected)")
        lines.append("")

        # Why This Is Risky
        lines.append(sep)
        lines.append("[bold]Why This Is Risky[/bold]")
        lines.append(sep)
        lines += [f"- {b}" for b in risky] if risky else ["- Existing safeguards"]
        lines.append("")

        # Optimization Guidance
        lines.append(sep)
        lines.append("[bold]Optimization Guidance (Safe / Do Not Touch)[/bold]")
        lines.append(sep)
        if safe:
            lines.append("[green]Safe:[/green]")
            lines += [f"- {s}" for s in safe]
        if do_not:
            lines.append("[bold red]DO NOT TOUCH:[/bold red]")
            lines += [f"‼ {d}" for d in do_not]
        lines.append("")

        # Brief Explanation
        lines.append(sep)
        lines.append("[bold]Brief Explanation[/bold]")
        lines.append(sep)
        lines.append(brief)

        console.print("\n".join(lines))
        # Minimal, neutral guidance footer (formatting only)
        console.print("[dim]Next step: restrict changes to non-functional optimizations only.[/dim]")
        console.print(f"\n[dim]Focus: {Path(c.file).name} — {c.risk_level} risk[/dim]\n")


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
