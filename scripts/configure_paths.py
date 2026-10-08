"""Configure legacy paper scripts once, retaining templates for safe reconfiguration."""
import argparse
import ast
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENS = ('REPO_ROOT', 'DATA_ROOT', 'MODEL_ROOT', 'OUTPUT_ROOT', 'CONDA_ROOT', 'CONDA_ENV')
SKIP = {'.git', 'Dataset', 'datasets', 'docs', 'assets', 'output', 'runs', '__pycache__', 'results'}

def configure(values, check=False):
    cache = ROOT / 'scripts/path_templates.json'
    templates = json.loads(cache.read_text()) if cache.exists() else {}
    for base, directories, names in os.walk(ROOT):
        directories[:] = [x for x in directories if x not in SKIP and not x.startswith(('output_', '.venv'))]
        for name in names:
            path = Path(base) / name
            if path.suffix not in {'.py', '.sh', '.yaml', '.yml'} or path.resolve() == Path(__file__).resolve():
                continue
            relative = str(path.relative_to(ROOT))
            text = path.read_text()
            if any(f'<{token}>' in text for token in TOKENS):
                templates.setdefault(relative, text)
    prepared = {}
    for relative, template in templates.items():
        text = template
        for token, value in values.items():
            text = text.replace(f'<{token}>', value)
        if relative.endswith('.py'):
            ast.parse(text, filename=relative)
        prepared[relative] = text
    if check:
        print(f'Validated configuration for {len(prepared)} files; no files changed.')
        return
    cache.write_text(json.dumps(templates, ensure_ascii=False, indent=2))
    for relative, text in prepared.items():
        path = ROOT / relative
        mode = path.stat().st_mode
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, encoding='utf-8', delete=False) as output:
            output.write(text)
            temporary = output.name
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    (ROOT / 'scripts/local_paths.json').write_text(json.dumps(values, indent=2))
    print(f'Configured {len(prepared)} files. Local templates and settings are ignored by git.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(ROOT / 'Dataset'))
    parser.add_argument('--model-root', required=True)
    parser.add_argument('--output-root', default=str(ROOT))
    parser.add_argument('--conda-root', default=str(Path(os.environ.get('CONDA_EXE', '/opt/conda/bin/conda')).parent.parent))
    parser.add_argument('--conda-env', default='trace')
    parser.add_argument('--check', action='store_true', help='Validate replacements without writing files.')
    args = parser.parse_args()
    values = dict(REPO_ROOT=str(ROOT), DATA_ROOT=str(Path(args.data_root).resolve()),
                  MODEL_ROOT=str(Path(args.model_root).resolve()), OUTPUT_ROOT=str(Path(args.output_root).resolve()),
                  CONDA_ROOT=str(Path(args.conda_root).resolve()), CONDA_ENV=args.conda_env)
    # Legacy shell launchers interpolate their configured paths into commands and heredocs.
    for token, value in values.items():
        if any(c in value for c in '\n\r\x00\"\x27`$\\') or any(c.isspace() for c in value) or any(c in value for c in ';&|<>()'):
            parser.error(f'{token} must not contain whitespace, quotes, or shell metacharacters.')
    configure(values, args.check)

if __name__ == '__main__':
    main()
