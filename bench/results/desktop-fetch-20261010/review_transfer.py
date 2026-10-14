"""Validate the two already-inspected pure functions in saved transfer responses.

Reject code outside a small AST subset; no imports, arbitrary attribute access,
while loops, builtins or top-level expressions can execute. Text quality remains
a human review, recorded separately in analysis.md.
"""
from pathlib import Path
import ast, hashlib, json, re

root = Path(__file__).resolve().parent
allowed = {ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Expr, ast.Constant,
           ast.If, ast.UnaryOp, ast.Not, ast.Return, ast.List, ast.Assign, ast.Name,
           ast.Store, ast.Load, ast.Subscript, ast.Slice, ast.For, ast.Compare, ast.Eq,
           ast.AugAssign, ast.Add, ast.Call, ast.Attribute, ast.Tuple, ast.ListComp,
           ast.comprehension, ast.GtE}
records = []
for path in sorted(root.glob('*/results.json')):
    data = json.loads(path.read_text(encoding='utf-8'))
    for row in data.get('transfer_workload', []):
        name = row['case']
        if name not in ('python-run-lengths', 'python-mutation-debug'):
            continue
        answer = row['response']['choices'][0]['message'].get('content') or ''
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', answer, re.S)
        record = {'run': path.parent.name, 'case': name, 'passed': False}
        try:
            if len(blocks) != 1:
                raise ValueError('Expected exactly one complete code fence')
            code = blocks[0]
            tree = ast.parse(code)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                raise ValueError('Expected one function only')
            for node in ast.walk(tree):
                if type(node) not in allowed:
                    raise ValueError('Unsupported AST: ' + type(node).__name__)
                if isinstance(node, ast.Name) and node.id.startswith('__'):
                    raise ValueError('Dunder name rejected')
                if isinstance(node, ast.Attribute) and node.attr != 'append':
                    raise ValueError('Only append attribute allowed')
                if isinstance(node, ast.Call) and not isinstance(node.func, ast.Attribute):
                    raise ValueError('Only append calls allowed')
            function = tree.body[0]
            expected_name = 'run_lengths' if name == 'python-run-lengths' else 'remove_negatives'
            if function.name != expected_name or function.decorator_list:
                raise ValueError('Unexpected function/decorator')
            namespace = {'__builtins__': {}}
            exec(compile(tree, '<reviewed-transfer>', 'exec'), namespace)
            cases = [([], []), ([3,3,1,3,3], [(3,2),(1,1),(3,2)]),
                     ([None,None,2], [(None,2),(2,1)]), ([1,2,2,1], [(1,1),(2,2),(1,1)])]
            if name == 'python-mutation-debug':
                cases = [([], []), ([2,-1,-3,4], [2,4]), ([0,-1,2,-3,0], [0,2,0])]
            for supplied, expected in cases:
                original = supplied.copy()
                actual = namespace[expected_name](supplied)
                if actual != expected or supplied != original or actual is supplied:
                    raise ValueError('Wrong result, input mutation or returned original list')
            record.update(passed=True, checks=len(cases), code_sha256=hashlib.sha256(code.encode()).hexdigest())
        except (ValueError, SyntaxError, KeyError, TypeError) as exc:
            record['error'] = str(exc)
        records.append(record)
(root/'transfer-code-checks.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
for record in records:
    print(record['run'], record['case'], 'PASS' if record['passed'] else record['error'])
