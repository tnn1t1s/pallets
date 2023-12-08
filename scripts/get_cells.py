import nbformat

thenb = 'nb/laebels.ipynb'

f = open(thenb)
notebook = nbformat.read(f, as_version=4)

#nb_fixed = nbformat.validator.normalize(nb_corrupted)
#nbformat.validator.validate(nb_fixed[1])

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        code = cell['source']
        print(code)
