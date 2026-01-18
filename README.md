# Bahmni / OpenMRS Risk‑Aware RAG Backend (CLI)

## What Problem This Solves (Legacy Risk & Uncertainty)
- Legacy clinical systems are brittle: changes can silently break core patient flows.
- Teams lack fast, grounded answers to: what rules exist, what’s risky, and what must not change.
- This backend reduces uncertainty by retrieving real code with attached risk signals for safer decisions.

## What the System Does (Risk‑Aware RAG Backend)
- Retrieves relevant Java code (e.g., `PatientService`) and injects impact metadata: risk level, score, dependencies, and defensive signals.
- Produces structured, enterprise‑safe output:
  - Critical Business Rules to Preserve
  - Why This Is Risky
  - Optimization Guidance (Safe / Do Not Touch)
- No code rewriting. No hallucination. No UI. Backend-only, grounded in sources.

## How to Run (CLI Command)
1) Install dependencies:

```bash
pip3 install -r requirements.txt
```

2) Optional: add the CLI to PATH for a neat `rag` command (macOS/zsh):

```bash
export PATH="$(pwd)/impact-cli:$PATH"
```

3) Ingest a small set of Java files (2–3 only) from Bahmni/OpenMRS:

```bash
python3 rag.py scan PatientService --root /path/to/openmrs-core
python3 rag.py ingest /path/to/openmrs-core/api/src/main/java/org/openmrs/api/PatientService.java --root /path/to/openmrs-core
```

4) Ask targeted questions (stable single result shown):

```bash
rag ask --best "Why is PatientService risky?"
rag ask --best "What business rules must be preserved here?"
rag ask --best "Is this service safe to optimize?"
```

If you didn’t add `rag` to PATH, use:

```bash
./impact-cli/rag ask --best "Why is PatientService risky?"
```

5) Interactive chat-style session (stateful, grounded, safe):

```bash
rag chat
# then ask multiple related questions:
# › Why is PatientService risky?
# › What must not change?
# › Is logging safe to optimize?
# Type 'exit' or 'quit' to end.
```

## Demo:
- Loom: https://www.loom.com/share/7195aed97d9d4c58ab1e5fe41fd887fe

## Future Plan (Backend First)
- Generalize the RAG engine across codebases with consistent risk signals.
- Improve structural and domain signals (transactions, authorization, exception contracts).
- Expose as a reusable backend service (JSON/HTTP), ready for CI/CD and tools.
