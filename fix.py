content = open('config.py', encoding='utf-8').read()

old = 'DB_PATH    = BASE_DIR / "gp_research.duckdb"'
new = 'DB_PATH    = BASE_DIR / "gp_output/gp_snapshot2.duckdb"'

if old in content:
    content = content.replace(old, new)
    open('config.py', 'w', encoding='utf-8').write(content)
    print('Fixed DB_PATH')
else:
    print('NOT FOUND')
    idx = content.find('DB_PATH')
    print(repr(content[idx:idx+100]))
