# Bahmni Impact & RAG CLI

Lightweight CLI to forecast change impact and risk in large Java codebases (starting with Bahmni / OpenMRS), plus a minimal RAG indexer to retrieve relevant code with impact context. It helps answer: "If I change this file or class, what could break?" and "Is this safe to optimize?"

## Features
- Impact analysis (impact.py):
  - Scans the codebase for files depending on the target class
  - Detects risk signals: throws/validation, transactional, null checks, domain-critical keywords
  - Computes an explainable risk score: LOW / MEDIUM / HIGH
  - Outputs the top affected files and signal breakdown
- RAG (rag.py):
  - Ingest 2–3 Java files, chunk them, and build local embeddings
  - Attach impact/risk metadata from `impact.py`
  - `ask` retrieves relevant code snippets with risk context and a brief grounded explanation

## Install
Python 3.9+ recommended. Install dependencies:

```bash
pip3 install -r requirements.txt
```

## Usage
### Impact analysis
Analyze a Java class and the downstream impact within a project root:

```bash
python3 impact.py path/to/PatientServiceImpl.java --root /path/to/bahmni-core
```

Show all dependent files instead of the top slice:

```bash
python3 impact.py path/to/PatientServiceImpl.java --root /path/to/bahmni-core --show-all
```

### RAG demo (index + ask)
1) Ingest a small set of Java files (2–3 files only):

```bash
python3 rag.py ingest path/to/PatientService.java path/to/PatientServiceImpl.java --root /path/to/bahmni-core
```

2) Ask targeted questions grounded in retrieved code + risk context:

```bash
python3 rag.py ask "Why is PatientService risky?"
python3 rag.py ask "What business rules must be preserved here?"
python3 rag.py ask "Is this service safe to optimize?"
```

## How it works (simplified)
### Impact
- Finds dependents via:
  - Direct imports: `import ... PatientServiceImpl;`
  - Indirect references: `new PatientServiceImpl(...)`, `extends PatientServiceImpl`, `PatientServiceImpl.method(...)`, etc.
- Scans the target file for signals:
  - `throw`/`throws`, `validate*`, `@Transactional`, null checks (`!= null`, `== null`, `not null`)
  - Domain keywords: patient, encounter, order, identifier, visit, provider, obs, concept, lab, drug
- Scores with simple weights prioritizing explainability over precision, then bins into LOW/MEDIUM/HIGH.

### RAG
- Chunks each Java file by lines (default 30, 8-line overlap) for readability.
- Builds lightweight embeddings using `HashingVectorizer` (local, fast, no model download).
- Attaches per-file risk metadata (`level`, `score`, `deps`) from `impact.py` to each chunk.
- Stores vectors in `.rag_index/vectors.npy` and metadata in `.rag_index/index.json`.
- `ask` embeds the question and returns top-k similar chunks with file name, risk level, relevant code snippet, and a brief explanation grounded in retrieved context.

## Notes & Limitations
- Text-based heuristics: No full semantic parsing, by design for speed.
- Package-level impacts and runtime wiring (Spring, transactions) are approximated via annotations/keywords.
- Excludes typical build output directories (`build`, `target`, `out`, `.git`, `node_modules`).
 - Embeddings use hashing-based vectors for the demo; swap in Sentence Transformers or FAISS later if needed.

## Roadmap
- Enrich domain keyword sets from Bahmni modules
- Optional exclusion of tests
- Support multi-class files and inner classes
- Export JSON for CI dashboards
- Optional FAISS or pgvector backend

## Contributing
PRs welcome. Keep changes small, focused, and explainable.
