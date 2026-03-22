content = open('main_train_gp.py', encoding='utf-8').read()

old = '''import sys, os'''

new = '''import sys, os
import random
import argparse
import numpy as np'''

if old in content and 'argparse' not in content:
    content = content.replace(old, new)
    print('Added imports')
else:
    print('imports already exist or pattern not found')

# Add seed parsing inside main()
old_main = '''def main():'''
new_main = '''def main():
    # Parse CLI arguments for reproducibility
    parser = argparse.ArgumentParser(description="GP Trading System Trainer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args, _ = parser.parse_known_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
'''

if old_main in content and 'parse_known_args' not in content:
    content = content.replace(old_main, new_main)
    open('main_train_gp.py', 'w', encoding='utf-8').write(content)
    print('Fixed #26 CLI seed')
else:
    print('Pattern not found or already fixed')
    idx = content.find('def main')
    print(repr(content[idx:idx+200]))
