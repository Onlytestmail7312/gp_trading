content = open('backtester.py', encoding='utf-8').read()

old = '''        if not in_trade:
            # Bull regime: only take LONG signals
            if curr_regime == 1:
                if prev_sig <= 0 and curr_sig > 0:
                    in_trade    = True
                    direction   = 1
                    entry_day   = i
                    entry_price = price * (1 + cost_pct)
                    entry_sig   = curr_sig
                    peak_price  = entry_price

            # Bear regime: only take SHORT signals
            elif curr_regime == -1:
                if prev_sig >= 0 and curr_sig < 0:
                    in_trade    = True
                    direction   = -1
                    entry_day   = i
                    entry_price = price * (1 - cost_pct)
                    entry_sig   = curr_sig
                    peak_price  = entry_price'''

new = '''        if not in_trade:
            # Bull regime: only take LONG signals
            if curr_regime == 1:
                if prev_sig <= 0 and curr_sig > 0 and curr_sig >= signal_threshold:
                    # Use next day open for realistic execution
                    next_open   = opens[i+1] if (opens is not None and i+1 < len(opens)) else price
                    in_trade    = True
                    direction   = 1
                    entry_day   = i
                    entry_price = next_open * (1 + cost_pct)
                    entry_sig   = curr_sig
                    peak_price  = entry_price

            # Bear regime: only take SHORT signals
            elif curr_regime == -1:
                if prev_sig >= 0 and curr_sig < 0 and abs(curr_sig) >= signal_threshold:
                    next_open   = opens[i+1] if (opens is not None and i+1 < len(opens)) else price
                    in_trade    = True
                    direction   = -1
                    entry_day   = i
                    entry_price = next_open * (1 - cost_pct)
                    entry_sig   = curr_sig
                    peak_price  = entry_price'''

if old in content:
    content = content.replace(old, new)
    open('backtester.py', 'w', encoding='utf-8').write(content)
    print('Fixed -- next day open entry')
else:
    print('Pattern not found')
    idx = content.find('if not in_trade')
    print(repr(content[idx:idx+300]))
