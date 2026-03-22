content = open('main_train_gp.py', encoding='utf-8').read()

old = 'log = get_logger()\n\n\ndef main():'
new = 'def main():'

if old in content:
    content = content.replace(old, new)
    # Add log inside main after seed setup
    content = content.replace(
        '    np.random.seed(args.seed)',
        '    np.random.seed(args.seed)\n    log = get_logger()'
    )
    open('main_train_gp.py', 'w', encoding='utf-8').write(content)
    print('Fixed')
else:
    print('Pattern not found')
    idx = content.find('log = get_logger')
    print(repr(content[idx-20:idx+100]))
