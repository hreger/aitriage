import json

with open('notebooks/advanced_triage_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'merged_df[numeric_cols] = imputer.fit_transform(merged_df[numeric_cols].values)' in source:
            # Add pd.DataFrame
            new_source = source.replace(
                'merged_df[numeric_cols] = imputer.fit_transform(merged_df[numeric_cols].values)',
                'merged_df[numeric_cols] = pd.DataFrame(imputer.fit_transform(merged_df[numeric_cols].values), columns=numeric_cols, index=merged_df.index)'
            )
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]
            break

with open('notebooks/advanced_triage_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
