#%% packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


#%% DSC mediaan, IQR en P10 berekenen.

csv_path = path...
out_dir = path...

filename = 'DSC_percentielen_A.csv'  
df = pd.read_csv(csv_path)

numeric_cols = df.select_dtypes(include='number').columns.tolist()
summary_stats = pd.DataFrame(columns=['Mediaan', 'Q1 (25%)', 'Q3 (75%)', 'IQR',
    'P10', 'P90', 'P10-90 Range',])


for col in numeric_cols:
    data = df[col].dropna()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    p10 = data.quantile(0.10)
    p90 = data.quantile(0.90)
    iqr_90 = p90 - p10

    summary_stats.loc[col] = [
        data.median(),
        q1,
        q3,
        iqr,
        p10,
        p90,
        iqr_90
    ]


print(summary_stats.round(3).to_string())
output_path = os.path.join(out_dir, filename)
summary_stats.to_csv(output_path)

#%% Boxplots voor visualisatie van DSC 

csv_path = 'C:/Users/jjkool/Documents/BEP/pythonscripts/statistiek/compositia/dsc_compo_b2_12juni.csv'

def boxplot(df, dataset_name):
    tissues = ['VAT', 'SM', 'SAT']
    counts = [df[t].dropna().shape[0] for t in tissues]
    labels = [f"{t} (n={c})" for t, c in zip(tissues, counts)]

    plt.figure(figsize=(8, 6))
    box = plt.boxplot([df[t].dropna() for t in tissues], tick_labels=labels, showmeans=True,
                      meanline=True, meanprops=dict(color='blue', linewidth=2, linestyle='--'),
                      medianprops=dict(color='orange', linewidth=2)) 
   
    #Legenda toevoegen
    plt.legend([box["medians"][0], box["means"][0]],
               ['Mediaan', 'Gemiddelde'],
               loc='lower right', frameon=True, fancybox=True, borderpad=1, fontsize = 12)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 1)
    plt.ylabel("Dice Similarity Coefficient", fontsize = 15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

df = pd.read_csv(csv_path)
dataset_name = 'CompositIA - subset A'
boxplot(df, dataset_name)
