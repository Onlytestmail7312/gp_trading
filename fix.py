content = open('gp_engine.py', encoding='utf-8').read()

old = '                    fitnesses = list(toolbox.map(toolbox.evaluate, chunk))'
new = '                    fitnesses = list(map(toolbox.evaluate, chunk))'

if old in content:
    content = content.replace(old, new)
    open('gp_engine.py', 'w', encoding='utf-8').write(content)
    print('Fixed -- using single-threaded evaluation')
else:
    print('Pattern not found')
    lines = content.split('\n')
    for i, l in enumerate(lines):
        if 'toolbox.map' in l or 'fitnesses' in l:
            print(f'Line {i}: {l}')
