#!/usr/bin/env python3

import pandas as pd
import numpy as np
import csv
import sys

filename = sys.argv[1]
outfilename = filename.replace('out', 'cv')

df = pd.read_table(filename, header=None,
                   names=['ggnl', 'gunl', 'ur', 'wr', 'w', 'sp', 'lr', 'bs', 'hs', 'ot', 'ml', 'fn',
                          'S', 'D', 'P', 'rZ', 'rT', 'rW', 'rV', 'sZ',
                          'sW', 'sV', 'sT', 'By', 'Acc'])

# Only taking rows with valid accuracy
df['Acc'].replace('', np.nan, inplace=True)
df.dropna(subset=['Acc'], inplace=True)

# Compute average accuracy, grouping by hyperparams
df_cv = df['Acc'].groupby([df['ggnl'], df['gunl'], df['ur'], df['wr'],
                           df['w'], df['sp'], df['lr'], df['bs'], df['hs'], df['ot'], df['ml'],
                           df['S'], df['D'], df['P'], df['rZ'], df['rT'], df['rW'],
                           df['rV'], df['sZ'],
                           df['sW'], df['sV'], df['sT'], df['By']
                           ]).mean().to_frame()

# Save to groupby file
df_cv.to_csv(outfilename, sep="\t", quoting=csv.QUOTE_NONE)

# Show best hyperparams
max = df_cv['Acc'].max()
idx = df_cv['Acc'].idxmax()

print('Best CV accuracy:', str(max))
print('Corresponding params')
print("\t".join([str(i) for i in ['ggnl', 'gunl', 'ur', 'wr', 'w', 'sp', 'lr', 'bs', 'hs', 'ot', 'ml',
                          'S', 'D', 'P', 'rZ', 'rT', 'rW', 'rV', 'sZ',
                          'sW', 'sV', 'sT', 'By']]))
print("\t".join([str(i) for i in idx]))
