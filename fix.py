import pandas as pd

df = pd.read_parquet('gp_output/gp_features_daily2.parquet')
print('Before - symbol dtype:', df['symbol'].dtype)

# Convert ArrowStringArray to regular string
df['symbol'] = df['symbol'].astype(str)
print('After  - symbol dtype:', df['symbol'].dtype)

# Verify
print('Sample:', df['symbol'].unique())

# Save back
df.to_parquet('gp_output/gp_features_daily2.parquet', index=True)
print('Saved with regular string dtype')

# Verify filter works
test = df[df['symbol'] == 'ICICIBANK']
print('ICICIBANK rows:', len(test))
