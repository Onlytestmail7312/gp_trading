content = open('config_v8.py', encoding='utf-8').read()

# Simply remove the print statement from validate_config
old = '''import os as _os
if _os.environ.get("WORKER_PROCESS") != "1":
    validate_config()
    print("  [OK] V8 config validation passed")'''

new = '''import os as _os
if _os.environ.get("WORKER_PROCESS") != "1":
    validate_config()'''

if old in content:
    content = content.replace(old, new)
    open('config_v8.py', 'w', encoding='utf-8').write(content)
    print('Fixed - removed print from workers')
else:
    # Try simpler approach - just suppress all prints in validate
    old2 = 'validate_config()\n    print("  [OK] V8 config validation passed")'
    new2 = 'validate_config()'
    if old2 in content:
        content = content.replace(old2, new2)
        open('config_v8.py', 'w', encoding='utf-8').write(content)
        print('Fixed v2')
    else:
        print('NOT FOUND - check config_v8.py manually')
        idx = content.find('validate_config')
        print(repr(content[idx:idx+300]))
