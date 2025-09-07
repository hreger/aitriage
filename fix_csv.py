import re

with open('data/diagnosis.csv', 'r') as f:
    lines = f.readlines()

header = lines[0].strip()
data_lines = lines[1:]

fixed_lines = [header]

for line in data_lines:
    parts = line.strip().split(',')
    if len(parts) > 6:
        first5 = ','.join(parts[:5])
        last = ','.join(parts[5:])
        fixed_line = f'{first5},"{last}"'
    else:
        fixed_line = line.strip()
    fixed_lines.append(fixed_line)

with open('data/diagnosis.csv', 'w') as f:
    f.write('\n'.join(fixed_lines))
