#!/usr/bin/env python

import sys
import shlex
import subprocess
import tempfile
import nbformat


def extract_code_cells(file_data):
    notebook = nbformat.reads(file_data, as_version=4)
    code = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell_code = cell['source']
            code.append(cell_code)
    return ''.join(code)


def run_cmd(cmd):
    args = shlex.split(cmd)
    result = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = result.communicate()
    return stdout, stderr


def code_from_git(filename, version='HEAD'):
    git_cmd = f'git cat-file -p {version}:{filename}'
    stdout, stderr = run_cmd(git_cmd)
    if stderr:
        msg = f'ERROR: could not read {filename} from commit {version}'
        raise Exception(msg)
    code = extract_code_cells(stdout)
    return code


def code_from_src(filename):
    src_data = open(filename).read()
    code = extract_code_cells(src_data)
    return code


def code_cmp(git_code, src_code):
    mk_tmp = tempfile.NamedTemporaryFile
    with mk_tmp() as git_f, mk_tmp() as src_f:
        git_f.write(git_code.encode())
        git_f.flush()
        src_f.write(src_code.encode())
        src_f.flush()
        stdout, stderr = run_cmd(f'diff {git_f.name} {src_f.name}')
        return stdout, stderr


if len(sys.argv) < 2:
    sys.stderr.write('ERROR: {sys.argv[0]} expected <filename> argument\n')
    sys.exit(-1)

filename = sys.argv[1]


git_code = code_from_git(filename)
src_code = code_from_src(filename)


stdout, stderr = code_cmp(git_code, src_code)
if stderr:
    raise Exception(stderr)
print(stdout)
