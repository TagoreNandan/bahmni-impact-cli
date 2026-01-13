# Bahmni Impact CLI

Lightweight CLI to forecast change impact and risk in large Java codebases (starting with Bahmni / OpenMRS). It answers: "If I change this file or class, what could break?"

## Features
- Scans the codebase for files depending on the target class
- Detects risk signals: throws/validation, transactional, null checks, domain-critical keywords
- Computes an explainable risk score: LOW / MEDIUM / HIGH
- Outputs the top affected files and signal breakdown

## Install
Python 3.9+ recommended. Install dependencies:

```bash
pip3 install typer rich
```

## Usage
Analyze a Java class and the downstream impact within a project root:

```bash
python3 impact.py path/to/PatientServiceImpl.java --root /path/to/bahmni-core
```

Show all dependent files instead of the top slice:

```bash
python3 impact.py path/to/PatientServiceImpl.java --root /path/to/bahmni-core --show-all
```

## How it works (simplified)
- Finds dependents via:
  - Direct imports: `import ... PatientServiceImpl;`
  - Indirect references: `new PatientServiceImpl(...)`, `extends PatientServiceImpl`, `PatientServiceImpl.method(...)`, etc.
- Scans the target file for signals:
  - `throw`/`throws`, `validate*`, `@Transactional`, null checks (`!= null`, `== null`, `not null`)
  - Domain keywords: patient, encounter, order, identifier, visit, provider, obs, concept, lab, drug
- Scores with simple weights prioritizing explainability over precision, then bins into LOW/MEDIUM/HIGH.

## Notes & Limitations
- Text-based heuristics: No full semantic parsing, by design for speed.
- Package-level impacts and runtime wiring (Spring, transactions) are approximated via annotations/keywords.
- Excludes typical build output directories (`build`, `target`, `out`, `.git`, `node_modules`).

## Roadmap
- Enrich domain keyword sets from Bahmni modules
- Optional exclusion of tests
- Support multi-class files and inner classes
- Export JSON for CI dashboards

## Contributing
PRs welcome. Keep changes small, focused, and explainable.
