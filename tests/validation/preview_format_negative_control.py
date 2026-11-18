"""Remove float32 vertex packing only in a disposable installed preview draw."""
import ast
import hashlib
import json
import pathlib
import sys


def mutate_preview_packing(source):
    """Change only draw()'s actual batch_for_shader position packing expression.

    The local geometry remains float64, and the native harness still invokes
    the installed draw callback. Its projected-pixel gate must detect a broken
    packing contract; no alternate renderer or fabricated failure is inserted.
    """
    tree=ast.parse(source)
    draws=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='draw']
    if len(draws)!=1:
        raise RuntimeError('Required preview draw function not reached exactly once')
    positions=[]
    for node in ast.walk(draws[0]):
        if not (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                and node.func.id=='batch_for_shader'):
            continue
        for argument in node.args:
            if isinstance(argument,ast.Dict):
                positions.extend(value for key,value in zip(argument.keys,argument.values)
                                 if isinstance(key,ast.Constant) and key.value=='pos')
    expected=ast.parse("np.ascontiguousarray(entry['points'], dtype=np.float32)",mode='eval').body
    if len(positions)!=1 or ast.dump(positions[0])!=ast.dump(expected):
        raise RuntimeError('Required float32 draw packing expression not reached exactly once')
    target=positions[0]
    encoded=source.encode('utf-8')
    lines=encoded.splitlines(keepends=True)
    start=sum(len(line) for line in lines[:target.lineno-1])+target.col_offset
    end=sum(len(line) for line in lines[:target.end_lineno-1])+target.end_col_offset
    removed=encoded[start:end].decode('utf-8')
    replacement="entry['points']"
    mutated=(encoded[:start]+replacement.encode('utf-8')+encoded[end:]).decode('utf-8')
    return mutated,removed,replacement


def main(path):
    path=pathlib.Path(path).resolve()
    if path.name!='authoring.py' or not any(part.startswith('aclw-') for part in path.parts):
        raise RuntimeError('Negative control is limited to a harness-owned installed profile')
    before=path.read_bytes()
    mutated,removed,replacement=mutate_preview_packing(before.decode('utf-8'))
    record={
        'purpose':'disposable negative control',
        'target':'authoring.draw / batch_for_shader / pos',
        'expression_removed':removed,
        'replacement':replacement,
        'before_sha256':hashlib.sha256(before).hexdigest(),
    }
    path.write_bytes(mutated.encode('utf-8'))
    record['after_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name('preview_format_mutation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('PREVIEW_FORMAT_MUTATION='+json.dumps(record))


if __name__=='__main__':
    main(sys.argv[1])
