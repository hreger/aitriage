import json

with open('notebooks/advanced_triage_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'numeric_cols = [col for col in numeric_cols if col in merged_df.columns]' in source:
            # Add unique
            new_source = source.replace(
                'numeric_cols = [col for col in numeric_cols if col in merged_df.columns]',
                'numeric_cols = [col for col in numeric_cols if col in merged_df.columns]\nnumeric_cols = list(set(numeric_cols))'
            )
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]
            break

with open('notebooks/advanced_triage_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
