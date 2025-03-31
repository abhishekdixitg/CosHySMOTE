import numpy as np
from sklearn.datasets import make_classification
from COSHysmote_V3 import COSHySMOTE
import pandas as pd
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')
from sklearn.model_selection import train_test_split

def relabel(df, column, search, result, test=False):
    print(30*"-")
    if test:
        print("Terms to relabel:")
    else:
        print(f"Terms relabeled to \"{result}\":")
    print(30*"-")
    print(df[df[column].str.contains(search)][column].value_counts())
    if test:
        return None
    else:
        df.loc[df[column].str.contains(search), column] = result

df = pd.read_csv('C:/Users/abhishekd/OneDrive - Adobe/Abhishek/Adobe_Oct_21/Abhishek Papers/Journal/Imbalance Data/COS-HySMOTE-Paper/HAM10000_dataset/unify/unify_diagnoses_example.csv', encoding='utf-8', delimiter=';', index_col=0)
df['dx'] = df.diagnosis.str.lower()
print(df.head())

# Relabel categories in the 'dx' column
relabel(df, "dx", "(unclear)|(no sign)|(no residual)|(scar)|(collis)", "nonuse")  # Nonuse cases
relabel(df, "dx", "(lentigo maligna)|(ssm)|(mela[n]*oma)", "mel")  # Melanoma
relabel(df, "dx", "(bcc)|(basal cell carcinom)", "bcc")  # Basal cell carcinoma
relabel(df, "dx", "(intraepithelial carc)|(bowen)|(actinic keratosis)", "akiec")  # AKIEC
relabel(df, "dx", "(n[ä|a]*e*vus(?! sebaceus))|(Compound)|(reed)", "nv")  # Nevus
relabel(df, "dx", "(seb k)|(verr.*seborrhoica)|(seborrh[o]*eic ker)", "bkl")  # Seborrheic keratosis
relabel(df, "dx", "angio(kerato)*m", "vasc")  # Vascular lesions
relabel(df, "dx", "dermatofibrom", "df")  # Dermatofibroma

# Retain only rows with the unified labels
df = df[df.dx.isin(['mel', 'nv', 'bcc', 'bkl', 'vasc', 'df', 'akiec'])]

# Display the updated class distribution
print("Class distribution after relabeling:")
print(df["dx"].value_counts())


# Preprocess X and y
X = df.drop(columns=["dx", "diagnosis"])  # Replace with actual non-feature columns
y = df["dx"].factorize()[0]  # Encode labels as integers


# Reset indices for X and y to align with class_indices
X = X.reset_index(drop=True)
y = pd.Series(y).reset_index(drop=True)

# Apply COSHySMOTE
coshysmote = COSHySMOTE(target_distribution={0: 10, 1: 20}, cluster_sizes={0: 3, 1: 5}, random_state=42)

try:
    X_resampled, y_resampled = coshysmote.fit_resample(X, y)
    print("Resampling successful. Class distribution after resampling:")
    print(pd.Series(y_resampled).value_counts())
except Exception as e:
    print("Error during resampling:", str(e))