#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from plot_util import *

# Add parent directory to path for model imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.model_smm_probe import *

def load_and_clean_data(csv_path):
    """Load and clean the CSV data"""
    df = pd.read_csv(csv_path)
    
    # Remove rows with missing GPU Time values
    df = df.dropna(subset=['GPU Time (sec)'])
    
    # Extract power from unique_keys (pow2) column for better x-axis labels
    df['power'] = np.log2(df['unique_keys'])
    df['unique_keys_log'] = np.log2(df['unique_keys'])
    df['build_num_keys'] = df['unique_keys'] * df['build_dup']
    df['build_num_keys_log'] = np.log2(df['build_num_keys'])

    if 'probe_num_keys' not in df.columns:
        df['probe_num_keys'] = df['unique_keys'] * df['probe_dup']

    df['probe_num_keys_log'] = np.log2(df['probe_num_keys'])

    df['GPU Time (ms)'] = df['GPU Time (sec)'].astype(float) * 1000
    
    return df

def create_clustered_bar_plot(df, **params):
    """Create a clustered bar plot showing GPU Time vs unique_keys for different probe_dup values"""
    
    # Filter data for build_dup = 1
    df_filtered = filter_df(df, **params, strict=True)
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.barplot(data=df_filtered, x='power', y='GPU Time (ms)', hue='probe_dup', ax=ax)
    
    # Get unique powers for x-axis
    powers = sorted(df_filtered['power'].unique())
    
    # Customize the plot
    ax.set_xlabel('Unique Keys (Powers of 2)', fontsize=12, fontweight='bold')
    ax.set_ylabel('GPU Time (milliseconds)', fontsize=12, fontweight='bold')
    # ax.set_title('GPU Performance vs Dataset Size for Different Probe Duplication Factors\n(build_dup=1)', 
    #              fontsize=14, fontweight='bold')
    
    # Set x-axis labels to show 2^x format
    ax.set_xticklabels([f'2^{power}' for power in powers])
    
    # Add legend
    ax.legend(title='Probe Duplication', loc='upper left')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    # ax.set_yscale('log')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    save_fig(fig, f"cuco_smm_pair_count_{dict_to_str(params)}", subdir="cuco_smm", suffix="pdf")

def plot_smm_build_sweep(df, **params):
    df = filter_df(df, **params, strict=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=df, x='unique_keys_log', y='GPU Time (ms)', hue="build_dup",  style = "build_dup", markers=True, markersize=8, ax=ax)
    ax.set_xlabel('Unique Keys (log2 scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('GPU Time (milliseconds)', fontsize=12, fontweight='bold')
    ax.set_xticks(np.unique(df['unique_keys_log']))
    # ax.set_xticklabels([f'2^{int(tick)}' for tick in ax.get_xticks()])
    ax.grid(True, alpha=0.3)
    save_fig(fig, f"cuco_smm_pair_count_build_sweep{dict_to_str(params)}", subdir="cuco_smm", suffix="pdf")


def plot_smm_build_sweep_model(df, **params):
    df = filter_df(df, **params, strict=True)
    df['Model GPU Time (ms)'] = df.apply(RowApplyer(predict_smm_probe_time_nvbench, nvbench_name_mapping), axis=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=df, x='unique_keys_log', y='Model GPU Time (ms)', hue="build_dup",  style = "build_dup", markers=True, markersize=8, ax=ax)
    ax.set_xlabel('Unique Keys (log2 scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('GPU Time (milliseconds)', fontsize=12, fontweight='bold')
    ax.set_xticks(np.unique(df['unique_keys_log']))
    # ax.set_xticklabels([f'2^{int(tick)}' for tick in ax.get_xticks()])
    ax.grid(True, alpha=0.3)
    save_fig(fig, f"cuco_smm_pair_count_build_sweep_model_{dict_to_str(params)}", subdir="cuco_smm", suffix="pdf")


def plot_smm_build_sweep_comparison(df, **params):
    df = filter_df(df, **params, strict=True)
    df['Model GPU Time (ms)'] = df.apply(RowApplyer(predict_smm_probe_time_nvbench, nvbench_name_mapping), axis=1)
    
    # Reshape data for combined plotting
    df_measured = df[['unique_keys_log', 'GPU Time (ms)', 'build_dup']].copy()
    df_measured['Source'] = 'Measured'
    df_measured = df_measured.rename(columns={'GPU Time (ms)': 'Time (ms)'})
    
    df_model = df[['unique_keys_log', 'Model GPU Time (ms)', 'build_dup']].copy()
    df_model['Source'] = 'Model'
    df_model = df_model.rename(columns={'Model GPU Time (ms)': 'Time (ms)'})
    
    df_combined = pd.concat([df_measured, df_model], ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=df_combined, x='unique_keys_log', y='Time (ms)', hue="Source", style="Source", markers=True, markersize=8, ax=ax)
    
    ax.set_xlabel('Unique Keys (log2 scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('GPU Time (milliseconds)', fontsize=12, fontweight='bold')
    ax.set_xticks(np.unique(df['unique_keys_log']))
    ax.grid(True, alpha=0.3)
    
    save_fig(fig, f"cuco_smm_pair_count_build_sweep_comparison_{dict_to_str(params)}", subdir="cuco_smm", suffix="pdf")


def create_heatmap(df, output_dir=None):
    """Create a heatmap showing GPU Time across different parameters"""
    
    # Filter data with valid GPU times
    df_filtered = df.dropna(subset=['GPU Time (sec)'])
    
    # Create pivot table for heatmap
    pivot_data = df_filtered.pivot_table(
        values='GPU Time (sec)', 
        index='unique_keys (pow2)', 
        columns=['build_dup', 'probe_dup'], 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(pivot_data, 
                annot=True, 
                fmt='.2e', 
                cmap='viridis', 
                cbar_kws={'label': 'GPU Time (seconds)'})
    
    plt.title('GPU Time Heatmap: Dataset Size vs Duplication Factors', 
              fontsize=14, fontweight='bold')
    plt.xlabel('(Build Duplication, Probe Duplication)', fontsize=12)
    plt.ylabel('Unique Keys', fontsize=12)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'cuco_smm_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'cuco_smm_heatmap.pdf'), 
                   bbox_inches='tight')
    
    plt.show()

def main():
    # Get the path to the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'data', 'cuco_static_multimap', 'smm_20251021_build_sweep.csv')
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Load and clean data
    print("Loading data...")
    df = load_and_clean_data(csv_path)
    
    # Create output directory
    
    # Create the main clustered bar plot
    # print("Creating clustered bar plot...")
    # create_clustered_bar_plot(df, build_dup=1, hit_ratio=1.0)
    # create_clustered_bar_plot(df, build_dup=2, hit_ratio=1.0)
    # create_clustered_bar_plot(df, build_dup=4, hit_ratio=1.0)

    # plot_smm_build_sweep(df, Key="I32")
    # plot_smm_build_sweep(df, Key="I64")

    # plot_smm_build_sweep_model(df, Key="I32", build_dup=1)
    # plot_smm_build_sweep_model(df, Key="I64", build_dup=1)

    # plot_smm_build_sweep(df, Key="I32", build_dup=1)
    # plot_smm_build_sweep(df, Key="I64", build_dup=1)

    plot_smm_build_sweep_comparison(df, Key="I32", build_dup=1)
    
    # Create additional heatmap visualization
    # print("Creating heatmap...")
    # create_heatmap(df, output_dir)

if __name__ == "__main__":
    main()
