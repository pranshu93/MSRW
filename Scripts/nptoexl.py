import numpy as np
import pandas as pd
arr = np.load("results.npy")
df = pd.DataFrame (arr)
filepath = 'my_excel_file.xlsx'
df.to_excel(filepath, index=False)

