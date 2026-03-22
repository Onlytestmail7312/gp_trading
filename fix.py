content = open('gp_engine.py', encoding='utf-8').read()

old = 'EarlyStopping(patience=5, min_delta=0.01)'
new = 'EarlyStopping(patience=GP_EARLY_STOP, min_delta=0.01)'

if old in content:
    content = content.replace(old, new)
    open('gp_engine.py', 'w', encoding='utf-8').write(content)
    print('Fixed -- now uses GP_EARLY_STOP =', end=' ')
    import re
    val = re.findall(r'GP_EARLY_STOP\s*=\s*\d+', open('config.py').read())
    print(val)
else:
    print('Pattern not found')
