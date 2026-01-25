import pandas as pd
from plot_util import *
import seaborn as sns
import os
import pathlib

def read_our_results(path):
    df = pd.read_csv(path)
    # Ensure time is valid
    df = df[df['GPU Time (sec)'].notna()]
    df['time (ms)'] = df['GPU Time (sec)'] * 1e3
    
    # We want consistent identification of "Ours"
    # plot_expr uses: df['algo'] = df['impl'].map({"fuse": "PTI", "nofuse": "BB"})
    # But here we probably just have "Ours" or verify if there are impl details.
    # The csv has "benchmark" not "impl". "Device" column exists.
    # We will just label it "Ours".
    df['algo'] = "Ours"
    
    # Make sure we have KeyType. Ours has KeyType.
    # Make sure we have num_rows. Ours has num_rows.
    return df

def read_cudf_results(path, **constants):
    df = pd.read_csv(path)
    for k, v in constants.items():
        df[k] = v
    
    # cudf csv has avg_time (in seconds based on values)
    df['time (ms)'] = df['avg_time'] * 1e3
    
    # Standardize KeyType
    # cudf has key_type: int32, int64
    type_map = {'int32': 'I32', 'int64': 'I64'}
    if 'key_type' in df.columns:
        df['KeyType'] = df['key_type'].map(type_map)
    
    df['algo'] = "cuDF"
    return df

def plot_time_vs_cols(df_in, **params):
    df = df_in.copy()
    
    # Filter
    df = filter_df(
        df, 
        **params, 
        must_have_attr=["num_rows", "KeyType"])
    
    if df.empty:
        print(f"No data for params: {params}")
        return

    # Create Label for X-axis: (K, P)
    # K = num_key_cols, P = num_nonkey_cols
    df['config_label'] = df.apply(
        lambda row: f"({row['num_key_cols']}, {row['num_nonkey_cols']})", axis=1
    )
    
    # Sort by keys then non-keys for consistent x-axis order
    df = df.sort_values(by=['num_key_cols', 'num_nonkey_cols'])
    
    # Create plot
    fig, ax = create_plot((6, 5))
    
    # Clustered bar plot
    # x = config_label
    # hue = algo
    ax = sns.barplot(
        data=df, 
        x='config_label', 
        y='time (ms)', 
        hue="algo", 
        hue_order=["Ours", "cuDF"], 
        ax=ax
    )
    
    # Legend
    ax.legend(title="", loc='best')
    
    # Adjust axes
    # Similar style to plot_expr
    adjust_ax(
        ax, 
        fontsize=18, 
        legend_fontsize=16, 
        title="", 
        xlabel = {"label": "(#KeyCols, #PayloadCols)"},
        ylabel = {"label": "Time (ms)", "rotation": 0, "x":0.1, "y":1.02},
        xticklabels = {"rotation": 0}
    )
    
    # Save figure
    # Filename should include parameters to distinguish
    save_fig(
        fig, 
        f"sort_time_vs_cols_{dict_to_str(params)}", 
        subdir=f"sort/{params.get('machine', '')}", 
        suffix="pdf"
    )

if __name__ == "__main__":
    workload = "sort"
    machines = ['a100', 'cscs'] # Add others if needed: 'a100', 'h100' based on directory structure
    resdir = pathlib.Path(__file__).parent.absolute() / ".."  / ".." / "exp_results"

    for machine in machines:
        try:
            # Read Ours
            # Pattern: sort_table_{machine}
            # Note: The file logic in plot_util searches for files starting with prefix
            # sort_table_cscs matches sort_table_cscs_20260122.csv
            ours_path = latest_result_file(resdir / machine / workload / f"sort_table_{machine}")
            ours = read_our_results(ours_path)
            
            # Read cuDF
            # Pattern: cudf_sort_table
            cudf_path = latest_result_file(resdir / machine / workload / "cudf_sort_table")
            cudf = read_cudf_results(cudf_path)
            
            # Combine
            df = pd.concat([ours, cudf], ignore_index=True)
            
            # Separate plots for KeyType and num_rows
            # Find unique values to iterate over
            # We intersect the values available in both (or just iterate over all and empty plots will normally appear if filter fails, but better to be safe)
            
            # Use 'ours' as the baseline for what we likely want to plot, or the combined 
            unique_rows = sorted(df['num_rows'].unique())
            unique_keys = sorted(df['KeyType'].dropna().unique())
            
            for num_rows in unique_rows:
                for KeyType in unique_keys:
                    # plot
                    plot_time_vs_cols(
                        df,
                        machine=machine,
                        num_rows=num_rows,
                        KeyType=KeyType,
                        num_nonkey_cols=2
                    )
                    
        except FileNotFoundError as e:
            print(f"Skipping machine {machine} due to missing file: {e}")
        except Exception as e:
            print(f"Error processing machine {machine}: {e}")
