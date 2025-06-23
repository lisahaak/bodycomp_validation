#%% packages
import pandas as pd
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
import os

#%% Shapiro-Wilk-toets


base_path =  path...
file_path = os.path.join(base_path, 'area_automatica_b2_12juni.csv')  #Pas aan voor subset
subset = 'b2'  

weefsels = ['VAT_manual_mm2', 'SM_manual_mm2', 'SAT_manual_mm2']
#weefsels = ['VAT_manual_hu', 'SM_manual_hu', 'SAT_manual_hu'] 


def shapiro_wilk_toets(file_path, subset, kolomnamen):
    df = pd.read_csv(file_path)
    resultaten = []

    for weefsel in weefsels:
        data = df[weefsel].dropna()

        if len(data) < 3:
            resultaten.append({
                'Subset': subset,
                'Weefsel': weefsel,
                'W-waarde': None,
                'P-waarde': None,
            })
            continue

        stat, p = shapiro(data)

        resultaten.append({
            'Subset': subset,
            'Weefsel': weefsel,
            'W-waarde': f"{stat:.3f}",
            'P-waarde': f"{p:.3f}"
        })

    return pd.DataFrame(resultaten)


resultaten_df = shapiro_wilk_toets(file_path, subset, weefsels)

print("\nShapiro-Wilk-toets resultaten:")
print(resultaten_df.to_string(index=False))


#%% Mann-Whitney-U-toets

base_path =  path...
df_b1 = pd.read_csv(os.path.join(base_path, 'dsc_automatica_b1_12juni.csv')) 

weefsels = ['VAT', 'SM', 'SAT']
alpha = 0.05

def mann_whitney_u_toets(df_b1, df_b2, weefsels):
    results = []

    for weefsel in weefsels:
        scores_b1 = df_b1[weefsel].dropna()
        scores_b2 = df_b2[weefsel].dropna()

        stat, p_value = mannwhitneyu(scores_b1, scores_b2, alternative='two-sided')

        results.append({
            'Weefsel': weefsel,
            'U-statistiek': round(stat, 3),
            'P-waarde': round(p_value, 4),
        })

    return pd.DataFrame(results)

resultaten_df = mann_whitney_u_toets(df_b1, df_b2, weefsels)

print("\nMann-Whitney-U-toets resultaten:")
print(resultaten_df.to_string(index=False))


