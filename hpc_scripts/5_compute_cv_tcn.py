#!/usr/bin/env python3

import pandas as pd
import numpy as np
import csv
import sys

filename = sys.argv[1]
outfilename = filename.replace('out', 'cv')

df = pd.read_table(filename, header=None,
                   names=['ksize', 'clip', 'levels', 'nhid', 'w', 'ml', 'dropout',
                          'sp', 'lr', 'ne', 'bs', 'hs', 'ot', 'fn', 'Acc'])

# Only taking rows with valid accuracy
df['Acc'].replace('', np.nan, inplace=True)
df.dropna(subset=['Acc'], inplace=True)

# Compute average accuracy, grouping by hyperparams
# Compute average accuracy, grouping by hyperparams
df_cv = df['Acc'].groupby([df['ksize'], df['clip'], df['levels'], df['nhid'],
                           df['w'], df['ml'], df['dropout'], df['sp'], df['lr'],
                           df['ne'], df['bs'], df['hs'], df['ot']]).mean().to_frame()

# Save to groupby file
df_cv.to_csv(outfilename, sep="\t", quoting=csv.QUOTE_NONE)

# Show best hyperparams
max = df_cv['Acc'].max()
idx = df_cv['Acc'].idxmax()

print('Best CV accuracy:', str(max))
print('Corresponding params')
print("\t".join([str(i) for i in ['ksize', 'clip', 'levels', 'nhid', 'w', 'ml', 'dropout',
                          'sp', 'lr', 'ne', 'bs', 'hs', 'ot']]))
print("\t".join([str(i) for i in idx]))
