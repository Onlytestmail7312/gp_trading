# fix_v5config.py - Apply V5 settings to config.py
content = open('config.py', encoding='utf-8').read()

replacements = [
    ('GP_POPULATION    = 5000', 'GP_POPULATION    = 7000'),
    ('GP_GENERATIONS   = 60',   'GP_GENERATIONS   = 80'),
    ('GP_CROSSOVER     = 0.75', 'GP_CROSSOVER     = 0.80'),
    ('GP_MUTATION      = 0.15', 'GP_MUTATION      = 0.12'),
    ('GP_ELITE         = 30',   'GP_ELITE         = 40'),
    ('GP_TOURNAMENT    = 7',    'GP_TOURNAMENT    = 9'),
    ('GP_MAX_DEPTH     = 8',    'GP_MAX_DEPTH     = 7'),
    ('GP_MAX_NODES     = 50',   'GP_MAX_NODES     = 40'),
    ('GP_EARLY_STOP    = 20',   'GP_EARLY_STOP    = 25'),
    ('GP_CHUNK_SIZE    = 300',  'GP_CHUNK_SIZE    = 350'),
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
print(f'\nTotal changes applied: {changed}')
