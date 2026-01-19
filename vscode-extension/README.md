# Legacy Risk Analyzer (VS Code)

Thin VS Code UI on top of the existing Python RAG backend for legacy-system risk insights.

## Command
- "Analyze Legacy Risk": runs on the active file and shows risk level, critical rules, why it’s risky, and optimization guardrails.

## Backend Contract
- Calls: `python rag.py analyze --file <abs> --json`
- Expects JSON fields: file, lines {start,end}, risk {level,score}, dependencies {direct,indirect}, critical_business_rules[], why_risky[], guidance {safe[], do_not_touch[]}, brief_explanation.

## Development
```bash
cd vscode-extension
npm install
npm run compile
# Press F5 in VS Code to launch Extension Development Host
```

Note: This extension does not modify code or provide refactors; it surfaces read-only, enterprise-safe insights.
