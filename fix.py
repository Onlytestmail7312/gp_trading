content = open('backtest_all_stocks.py', encoding='utf-8').read()

old = '''            if curr_regime == 1 and prev_sig <= 0 and curr_sig > 0:
                in_trade    = True
                direction   = 1
                entry_day   = i
                entry_price = price * (1 + COST)
                peak_price  = entry_price

            elif curr_regime == -1 and prev_sig >= 0 and curr_sig < 0:
                in_trade    = True
                direction   = -1
                entry_day   = i
                entry_price = price * (1 - COST)
                peak_price  = entry_price'''

new = '''            if curr_regime == 1 and prev_sig <= 0 and curr_sig > 0:
                in_trade    = True
                direction   = 1
                entry_day   = i
                entry_price = price * (1 + COST)
                entry_sig   = curr_sig
                peak_price  = entry_price

            elif curr_regime == -1 and prev_sig >= 0 and curr_sig < 0:
                in_trade    = True
                direction   = -1
                entry_day   = i
                entry_price = price * (1 - COST)
                entry_sig   = curr_sig
                peak_price  = entry_price'''

if old in content:
    content = content.replace(old, new)
    open('backtest_all_stocks.py', 'w', encoding='utf-8').write(content)
    print('Fixed -- entry_sig now set at entry point')
else:
    print('Pattern not found')
    idx = content.find('if curr_regime == 1 and prev_sig')
    print(repr(content[idx:idx+300]))
