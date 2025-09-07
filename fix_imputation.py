import json

with open('notebooks/advanced_triage_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'merged_df[numeric_cols] = imputer.fit_transform(merged_df[numeric_cols])' in source:
            # Add .values
            new_source = source.replace(
                'merged_df[numeric_cols] = imputer.fit_transform(merged_df[numeric_cols])',
                'merged_df[numeric_cols] = imputer.fit_transform(merged_df[numeric_cols].values)'
            )
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]
            break

with open('notebooks/advanced_triage_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
