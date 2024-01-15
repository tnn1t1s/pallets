#!/usr/bin/env python

import sys
import nbformat


if len(sys.argv) < 2:
    sys.stderr.write('ERROR: {sys.argv[0]} expected <filename> argument\n')
    sys.exit(-1)

filename = sys.argv[1]

f = open(filename)
notebook = nbformat.read(f, as_version=4)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        code = cell['source']
        print(code)
