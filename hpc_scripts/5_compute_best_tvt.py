#!/usr/bin/env python3

import pandas as pd
import numpy as np
import csv
import sys

filename = sys.argv[1]
outfilename = filename.replace('out', 'cv')

df = pd.read_table(filename, header=None,
                   names=['ggnl', 'gunl', 'ur', 'wr', 'w', 'sp', 'lr', 'bs', 'hs', 'ot', 'ml', 'Tr_Acc', 'Val_Acc', 'Acc'])

# Only taking rows with valid accuracy
df['Acc'].replace('', np.nan, inplace=True)
df.dropna(subset=['Acc'], inplace=True)

# Compute average accuracy, grouping by hyperparams
#df_cv = df['Acc'].groupby([df['ggnl'], df['gunl'], df['ur'], df['wr'],
#                           df['w'], df['sp'], df['lr'], df['bs'], df['hs'], df['ot'], df['ml']]).mean().to_frame()

# Save to groupby file
#df_cv.to_csv(outfilename, sep="\t", quoting=csv.QUOTE_NONE)

# Show best hyperparams
max = df['Acc'].max()
idx = df.loc[df['Acc'].idxmax()].tolist()

print('Best Test accuracy:', str(max))
print('Corresponding params')
print("\t".join([str(i) for i in ['ggnl', 'gunl', 'ur', 'wr', 'w', 'sp', 'lr', 'bs', 'hs', 'ot', 'ml']]))
print("\t".join([str(i) for i in idx]))

# Create rerun string for best hyperparams
param_str = ['-ggnl', '-gunl', '-ur', '-wr', '-w', '-sp', '-lr', '-bs', '-hs', '-ot', '-ml']
print('Best hyperparam string')
print(" ".join([str(item) for sublist in list(map(list,zip(param_str,idx))) for item in sublist]))
