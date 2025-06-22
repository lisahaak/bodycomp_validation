#%% packages

import pandas as pd
import matplotlib.pyplot as plt
import os


#%% Bland-Altman-Analyse

def bland_altman_plot(ax, data, manual_col, auto_col, tissue_type, model, dataset, title_label, y_limits=None, x_limits=None, unit_label=""):
    avg = (data[manual_col] + data[auto_col]) / 2
    diff = data[auto_col] - data[manual_col]
    mean_diff = diff.mean()
    std_diff = diff.std()

    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    ax.scatter(avg, diff, color='blue', alpha=0.5)
    ax.axhline(mean_diff, color='gray', linestyle='--', label=f'Gemiddeld verschil = {mean_diff:.2f}{unit_label}')
    ax.axhline(loa_upper, color='orange', linestyle='--', label=f'Bovengrens LoA = {loa_upper:.2f}{unit_label}')
    ax.axhline(loa_lower, color='orange', linestyle='--', label=f'Ondergrens LoA = {loa_lower:.2f}{unit_label}')

    ax.set_title(f'{tissue_type}', fontsize=15)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=14)  

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}"))

    if y_limits:
        ax.set_ylim(y_limits)
    if x_limits:
        ax.set_xlim(x_limits)

    ax.legend(fontsize = 12)


def run_bland_altman_plot(csv_path, value_suffix, model, dataset, title_label, y_label, x_label, output_dir=None):
    df = pd.read_csv(csv_path)

    convert_to_cm2 = value_suffix.lower() in ["area", "oppervlakte", "mm2"]
    scale_factor = 0.01 if convert_to_cm2 else 1.0
    unit_label = " cm²" if convert_to_cm2 else " hu"

    desired_order = ['VAT', 'SM', 'SAT']
    all_tissues = {col.replace(f"_manual_{value_suffix}", "") for col in df.columns if col.endswith(f"_manual_{value_suffix}")}
    tissues = [t for t in desired_order if t in all_tissues]

    global_ymin, global_ymax = float('inf'), float('-inf')
    global_xmin, global_xmax = float('inf'), float('-inf')
    subset_data = {}

    for tissue in tissues:
        manual_col = f"{tissue}_manual_{value_suffix}"
        auto_col = f"{tissue}_auto_{value_suffix}"
        if manual_col in df.columns and auto_col in df.columns:
            subset = df[[manual_col, auto_col]].dropna().copy()
            if not subset.empty:
                subset[manual_col] *= scale_factor
                subset[auto_col] *= scale_factor

                avg = (subset[manual_col] + subset[auto_col]) / 2
                diff = subset[auto_col] - subset[manual_col]
                global_ymin = min(global_ymin, diff.min())
                global_ymax = max(global_ymax, diff.max())
                global_xmin = min(global_xmin, avg.min())
                global_xmax = max(global_xmax, avg.max())
                subset_data[tissue] = subset

    y_padding = 0.05 * (global_ymax - global_ymin) if global_ymax > global_ymin else 1
    x_padding = 0.05 * (global_xmax - global_xmin) if global_xmax > global_xmin else 1
    y_limits = (global_ymin - y_padding, global_ymax + y_padding)
    x_limits = (global_xmin - x_padding, global_xmax + x_padding)

    fig, axes = plt.subplots(1, len(tissues), figsize=(7 * len(tissues), 7), sharex=True, sharey=True)
    if len(tissues) == 1:
        axes = [axes]

    for i, tissue in enumerate(tissues):
        ax = axes[i]
        data = subset_data[tissue].rename(columns={
            f"{tissue}_manual_{value_suffix}": "manual",
            f"{tissue}_auto_{value_suffix}": "auto"
        })
        bland_altman_plot(ax, data, "manual", "auto", tissue, model, dataset, title_label, y_limits, x_limits, unit_label=unit_label)

        if i == 0:
            ax.set_ylabel(y_label, fontsize=15)
        else:
            ax.set_ylabel("")

    fig.supxlabel(x_label, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 1])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        safe_model = model.replace(" ", "_")
        safe_dataset = dataset.replace(" ", "_")
        safe_title = title_label.replace(" ", "_")

        filename = f"BA_plot_{safe_model}_{safe_dataset}_{safe_title}.png"
        full_path = os.path.join(output_dir, filename)
        fig.savefig(full_path, dpi=300)
        print(f"Plot opgeslagen als: {full_path}")

    plt.show()
    plt.close()


#%% Automatica oppervlakte

run_bland_altman_plot(
    csv_path = path...,
    model='AutoMATiCA',
    dataset='B2',
    value_suffix='mm2',
    title_label='oppervlakte',
    y_label='Verschil in oppervlakte (automatisch - handmatig) (cm$^2$)', 
    x_label='Gemiddelde oppervlakte van handmatige en automatische segmentatie (cm$^2$)',
    output_dir = path...
)

#%% Compositia oppervlakte

run_bland_altman_plot(
    csv_path = path...,
    model='CompositIA',
    dataset='B2',
    value_suffix='mm2',
    title_label='oppervlakte',
    y_label='Verschil in oppervlakte (automatisch - handmatig) (cm$^2$)', 
    x_label='Gemiddelde oppervlakte van handmatige en automatische segmentatie (cm$^2$)',
    output_dir = path...
)

#%% Automatica weefseldichtheid

run_bland_altman_plot(
    csv_path = path...,
    model='AutoMATiCA',
    dataset='B2',
    value_suffix='hu',
    title_label='weefseldichtheid',
    y_label='Verschil in weefseldichtheid (automatisch - handmatig) (HU)',
    x_label='Gemiddelde weefseldichtheid van handmatige en automatische segmentatie (HU)',
    output_dir = path...
)

#%% Compositia weefseldichtheid 

run_bland_altman_plot(
    csv_path = path ...,
    model='CompositIA',
    dataset='B2', 
    value_suffix='hu',
    title_label='weefseldichtheid',
    y_label='Verschil in weefseldichtheid (automatisch - handmatig) (HU)',
    x_label='Gemiddelde weefseldichtheid van handmatige en automatische segmentatie (HU)',
    output_dir = path...
)
