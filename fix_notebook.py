import json

with open('notebooks/triage_model.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'merged_df[numeric_cols] = imputer.fit_transform(merged_df[numeric_cols])' in source:
            # Add the line before
            new_source = source.replace(
                '# KNN imputation for missing values\nimputer = KNNImputer(n_neighbors=5)',
                '# KNN imputation for missing values\nnumeric_cols = [col for col in numeric_cols if col in merged_df.columns]\nimputer = KNNImputer(n_neighbors=5)'
            )
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]
            break

with open('notebooks/triage_model.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
