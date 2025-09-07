import json

with open('notebooks/advanced_triage_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'merged_df[numeric_cols] = imputer.fit_transform(merged_df[numeric_cols])' in source:
            # Add the line before
            new_source = source.replace(
                'numeric_cols = [col for col in numeric_cols if col in merged_df.columns]\nimputer = KNNImputer(n_neighbors=5)\nmerged_df[numeric_cols] = imputer.fit_transform(merged_df[numeric_cols])',
                'numeric_cols = [col for col in numeric_cols if col in merged_df.columns]\nif numeric_cols:\n    imputer = KNNImputer(n_neighbors=5)\n    merged_df[numeric_cols] = imputer.fit_transform(merged_df[numeric_cols])'
            )
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]
            break

with open('notebooks/advanced_triage_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
