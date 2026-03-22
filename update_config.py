content = open('config.py', encoding='utf-8').read()

replacements = [
    ('GP_POPULATION    = 2000', 'GP_POPULATION    = 3000'),
    ('GP_CROSSOVER     = 0.7',  'GP_CROSSOVER     = 0.75'),
    ('GP_MUTATION      = 0.2',  'GP_MUTATION      = 0.15'),
    ('GP_ELITE         = 10',   'GP_ELITE         = 20'),
    ('GP_TOURNAMENT    = 5',    'GP_TOURNAMENT    = 7'),
    ('GP_MAX_DEPTH     = 6',    'GP_MAX_DEPTH     = 8'),
    ('GP_MAX_NODES     = 40',   'GP_MAX_NODES     = 50'),
    ('GP_EARLY_STOP    = 15',   'GP_EARLY_STOP    = 20'),
    ('GP_CHUNK_SIZE    = 200',  'GP_CHUNK_SIZE    = 300'),
]

changed = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f'Changed: {old.strip()} -> {new.strip()}')
        changed += 1
    else:
        print(f'NOT FOUND: {old.strip()}')

open('config.py', 'w', encoding='utf-8').write(content)
print(f'\nTotal changes: {changed}')
