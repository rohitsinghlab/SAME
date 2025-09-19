import pandas as pd
import numpy as np
import anndata as ad


def createSAMEInput(inAD, x_col = 'array_row', y_col = 'array_col', ctype_col = 'class',
 onehot=True, zero_center=False):
    inDF = inAD.obs.copy()
    # Convert ctype_col to one-hot encoding and add to inDF
    if onehot:
        onehot = inDF[[ctype_col]].copy()
        onehot = onehot.astype(str)
        onehot_df = onehot[ctype_col].str.get_dummies()
    else:
        # If not onehot, check if unique values in ctype_col are columns in inDF and use that instead
        unique_classes = inDF[ctype_col].unique()
        if all(str(cls) in inDF.columns for cls in unique_classes):
            onehot_df = inDF[[str(cls) for cls in unique_classes]].copy()
        else:
            raise ValueError("One-hot columns for all unique values in ctype_col are not present in inDF.")
    # Add one-hot columns to inDF, prefix with 'class_'
    inDF = inDF.join(onehot_df)
    inDF = inDF[[x_col, y_col, ctype_col, *onehot_df.columns]]
    inDF.columns = ['X', 'Y', 'Cell Type', *onehot_df.columns]
    inDF.loc[:,'Cell_Num_Old'] = inDF.index.values
    inDF = inDF.reset_index(drop=True)
    if zero_center:
        inDF[['X', 'Y']] = inDF[['X', 'Y']] - inDF[['X', 'Y']].mean(axis=0)
    return inDF