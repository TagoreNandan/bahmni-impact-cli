import * as vscode from 'vscode';
import { execFile } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

function stripAnsi(input: string): string {
  // Remove ANSI escape codes (colors, styles) that may be added by rich
  // eslint-disable-next-line no-control-regex
  const ansiRegex = /\u001b\[[0-9;]*m/g;
  return input.replace(ansiRegex, '');
}

function runBackendAnalyze(backendRoot: string, filePath: string, projectRoot: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const py = process.env.PYTHON || 'python3';
    const args = ['rag.py', 'analyze', '--file', filePath, '--json', '--root', projectRoot];
    const options = { cwd: backendRoot };

    execFile(py, args, options, (error, stdout, stderr) => {
      if (error) {
        reject(new Error(stderr || error.message));
        return;
      }
      try {
        const raw = stdout.toString();
        const clean = stripAnsi(raw).trim();
        const json = JSON.parse(clean);
        resolve(json);
      } catch (e: any) {
        reject(new Error(`Failed to parse backend JSON: ${e.message}\nOutput: ${stdout}\nErrors: ${stderr}`));
      }
    });
  });
}

function renderHtml(result: any): string {
  const riskLevel: string = result?.risk?.level || 'UNKNOWN';
  const riskScore: number = result?.risk?.score ?? 0;
  const color = riskLevel === 'HIGH' ? '#b00020' : riskLevel === 'MEDIUM' ? '#cc7a00' : '#1b5e20';

  const rules: string[] = result?.critical_business_rules || [];
  const why: string[] = result?.why_risky || [];
  const safe: string[] = result?.guidance?.safe || [];
  const dnt: string[] = result?.guidance?.do_not_touch || [];

  const sep = '<hr class="sep" />';
  const bullets = (items: string[], emph = false, symbol = '•') =>
    items.map((i) => `<li${emph ? ' class="emph"' : ''}>${symbol === '•' ? '' : `<span class="sym">${symbol}</span>`}${i}</li>`).join('');

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>
  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; padding: 12px 16px; }
  h1 { font-size: 14px; margin: 0 0 8px; }
  .verdict { font-weight: 700; color: ${color}; margin-bottom: 8px; }
  .meta { color: #666; margin-bottom: 8px; }
  .label { font-weight: 600; }
  .sep { border: 0; border-top: 1px solid #ddd; margin: 12px 0; }
  h2 { font-size: 12px; margin: 0 0 8px; color: #333; }
  ul { margin: 0 0 8px 14px; padding: 0; }
  li { margin: 4px 0; }
  li.emph { color: #b00020; font-weight: 600; }
  .sym { margin-right: 6px; color: #b00020; font-weight: 700; }
  .footer { color: #999; font-size: 11px; margin-top: 8px; }
</style>
<title>Legacy Risk Insights</title>
</head>
<body>
  <div class="verdict">Verdict: ${riskLevel === 'HIGH' ? 'BEHAVIORAL OPTIMIZATION IS NOT SAFE' : riskLevel === 'MEDIUM' ? 'CAUTION: AVOID BEHAVIORAL CHANGES' : 'SAFE FOR NON-FUNCTIONAL OPTIMIZATIONS ONLY'}</div>
  <div class="meta"><span class="label">File:</span> ${result?.file || ''}</div>
  <div class="meta"><span class="label">Lines:</span> ${result?.lines?.start ?? ''}–${result?.lines?.end ?? ''}</div>
  <div class="meta"><span class="label">Risk:</span> ${riskLevel} (score ${riskScore})</div>
  <div class="meta"><span class="label">Dependencies:</span> direct=${result?.dependencies?.direct ?? 0}, indirect=${result?.dependencies?.indirect ?? 0}</div>

  ${sep}
  <h2>Critical Business Rules to Preserve</h2>
  ${rules.length ? `<ul>${bullets(rules)}</ul>` : '<div>None detected</div>'}

  ${sep}
  <h2>Why This Is Risky</h2>
  <ul>${bullets(why)}</ul>

  ${sep}
  <h2>Optimization Guidance</h2>
  ${safe.length ? '<div><strong>Safe:</strong><ul>' + bullets(safe) + '</ul></div>' : ''}
  ${dnt.length ? '<div><strong>DO NOT TOUCH:</strong><ul>' + bullets(dnt, true, '‼') + '</ul></div>' : ''}

  ${sep}
  <h2>Brief Explanation</h2>
  <div>${result?.brief_explanation || ''}</div>

  <div class="footer">Next step: restrict changes to non-functional optimizations only.</div>
</body>
</html>`;
}

export function activate(context: vscode.ExtensionContext) {
  // Lightweight signal to confirm the extension is loaded in the Dev Host
  vscode.window.setStatusBarMessage('Legacy Risk Analyzer active', 3000);

  const disposable = vscode.commands.registerCommand('legacyRisk.analyze', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage('Open a file to analyze.');
      return;
    }

    const filePath = editor.document.uri.fsPath;
    const workspaceFolders = vscode.workspace.workspaceFolders;
    const fileWorkspace = vscode.workspace.getWorkspaceFolder(editor.document.uri);
    const projectRoot = fileWorkspace ? fileWorkspace.uri.fsPath : (workspaceFolders && workspaceFolders.length ? workspaceFolders[0].uri.fsPath : '');

    // Determine backend root (where rag.py lives)
    const config = vscode.workspace.getConfiguration('legacyRisk');
    let backendRoot = String(config.get('backendRoot') || '').trim();
    if (!backendRoot) {
      backendRoot = (workspaceFolders && workspaceFolders.length) ? workspaceFolders[0].uri.fsPath : '';
    }
    const ragPath = backendRoot ? path.join(backendRoot, 'rag.py') : '';
    if (!backendRoot || !fs.existsSync(ragPath)) {
      vscode.window.showErrorMessage('Backend not found. Set "legacyRisk.backendRoot" to the folder that contains rag.py.');
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      'legacyRiskInsights',
      'Legacy Risk Insights',
      vscode.ViewColumn.Beside,
      { enableScripts: true }
    );

    panel.webview.html = `<html><body>Running analysis…</body></html>`;

    try {
      const result = await runBackendAnalyze(backendRoot, filePath, projectRoot);
      panel.webview.html = renderHtml(result);
    } catch (err: any) {
      panel.webview.html = `<html><body><pre>${(err?.message || String(err)).replace(/</g, '&lt;')}</pre></body></html>`;
    }
  });

  context.subscriptions.push(disposable);
}

export function deactivate() {}
