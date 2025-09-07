import nbformat

nb = nbformat.read('notebooks/advanced_triage_model.ipynb', as_version=4)
nbformat.write(nb, 'notebooks/advanced_triage_model.ipynb')
