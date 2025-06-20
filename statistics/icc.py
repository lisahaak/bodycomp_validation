#%%packages 
import pandas as pd
import pingouin as pg

#%% ICC berekenen

def icc3(csv_path, value_suffix='hu'):

    manual_suffix = f"_manual_{value_suffix}"
    auto_suffix = f"_auto_{value_suffix}"

    df = pd.read_csv(csv_path)

    weefsels = [col.replace(manual_suffix, "")
               for col in df.columns if col.endswith(manual_suffix)]

    results = []

    for weefsel in weefsels:
        manual_col = f"{weefsel}{manual_suffix}"
        auto_col = f"{weefsel}{auto_suffix}"

        if manual_col in df.columns and auto_col in df.columns:
            df_tissue = df[["scan_id", manual_col, auto_col]].dropna()

            # Zet om naar long format
            df_long = pd.melt(df_tissue, id_vars="scan_id",
                              value_vars=[manual_col, auto_col],
                              var_name="rater", value_name="score")

            df_long["rater"] = df_long["rater"].map({
                manual_col: "manual",
                auto_col: "auto"
            })

            # Bereken ICC(3,1)
            icc = pg.intraclass_corr(data=df_long,
                                     targets="scan_id",
                                     raters="rater",
                                     ratings="score")
            icc_3_1 = icc[icc["Type"] == "ICC3"].copy()
            icc_3_1["weefsel"] = weefsel

            results.append(icc_3_1)

    if results:
        result_df = pd.concat(results, ignore_index=True)
        print(result_df[["weefsel", "ICC", "CI95%", "F", "pval"]])
        return result_df
    else:
        print("Geen weefsels met geldige kolommen gevonden.")
        return pd.DataFrame()


#%% oppervlakte 
csv_file = 'C:/Users/jjkool/Documents/BEP/pythonscripts/statistiek/compositia/area_compo_a_12juni.csv' #Pas aan voor subset
#csv_file = 'C:/Users/jjkool/Documents/BEP/pythonscripts/statistiek/automatica/area_automatica_a_12juni.csv'

value_suffix = 'mm2'
icc3_result = icc3(csv_file, value_suffix)

#%% weefseldichtheid

csv_file = 'C:/Users/jjkool/Documents/BEP/pythonscripts/statistiek/compositia/hu_compo_a_12juni.csv' #Pas aan voor subset
#csv_file = 'C:/Users/jjkool/Documents/BEP/pythonscripts/statistiek/automatica/hu_automatica_a_12juni.csv'

value_suffix = 'hu'
icc3_result = icc3(csv_file)


