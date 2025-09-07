import json

with open('notebooks/advanced_triage_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'metadata' in cell:
        if not isinstance(cell['metadata'], dict):
            cell['metadata'] = {}
        if 'trusted' in cell['metadata']:
            del cell['metadata']['trusted']

with open('notebooks/advanced_triage_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
