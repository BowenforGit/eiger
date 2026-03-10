import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_util import *
import pathlib
import numpy as np

def plot_hashing_performance(csv_path, machine, **params):
    df = pd.read_csv(csv_path)
    df = df[df['GPU Time (sec)'].notna()]
    df = filter_df(
        df, 
        **params, 
        strict=True, 
        must_have_attr=["read_input", "reduce", "HashFunction"])
    df['throughput_GBps'] = df['InputBytes (bytes)'] / df['GPU Time (sec)'] / 1e9
    df['num_items_log2'] = np.log2(df['num_items'])

    fig, ax = create_plot((8,6))
    sns.lineplot(data=df, x='num_items_log2', y='throughput_GBps', hue='KeyType', style = 'KeyType', markers=True, markersize=8,ax=ax)
    ax.set_xticks(np.unique(df['num_items_log2']))

    adjust_ax(
        ax,
        fontsize=24,
        legend_fontsize=20,
        title=f"{params['HashFunction']} Performance",
        xlabel={"label": "Number of Items (log2 Scale)", "padding": 5},
        ylabel={"label": "Throughput (GB/s)"}
    )
    
    plt.tight_layout()
    save_fig(fig, f"hashing_{dict_to_str(params)}", subdir=f"archim/{machine}", suffix="pdf")

def plot_gather_performance(csv_path, machine, **params):
    df = pd.read_csv(csv_path)
    df = df[df['GPU Time (sec)'].notna()]
    df = filter_df(
        df, 
        **params, 
        strict=True)
    df['throughput_GBps'] = (df['InputRead (bytes)'] + df['OutputWrite (bytes)'] + df['GatherMapRead (bytes)']) / df['GPU Time (sec)'] / 1e9
    df['num_inputs_log2'] = np.log2(df['num_inputs'])

    fig, ax = create_plot((12,6))
    sns.lineplot(data=df, x='num_inputs_log2', y='throughput_GBps', hue='ValueType', style = 'ValueType', markers=True, markersize=8,ax=ax)
    ax.set_xticks(np.unique(df['num_inputs_log2'])[0::2])

    adjust_ax(
        ax,
        fontsize=24,
        legend_fontsize=20,
        title=f"Gather Performance",
        xlabel={"label": "Number of Items (log2 Scale)", "padding": 5},
        ylabel={"label": "Throughput (GB/s)"}
    )
    
    plt.tight_layout()
    save_fig(fig, f"gather_{dict_to_str(params)}", subdir=f"archim/{machine}", suffix="pdf")

def plot_sort_pair_performance(csv_path, machine, **params):
    df = pd.read_csv(csv_path)
    df = df[df['GPU Time (sec)'].notna()]
    df = filter_df(
        df, 
        **params, 
        strict=True)
    df['passes'] = df['num_bits'] / 8
    df['passes'] = df['passes'].astype(int)
    df['bytes_per_item'] = df['Key'].apply(lambda x: 4 if x == 'I32' else 8)
    df['time_per_pass_msec'] = df['GPU Time (sec)'] / df['passes'] * 1e3
    df['throughput_per_pass_GBps'] = (df['bytes_per_item'] * df['num_items'] * 4 * df['passes']) / df['GPU Time (sec)'] /  1e9
    df['num_items_log2'] = np.log2(df['num_items'])
    
    fig, ax1 = create_plot((8,6))

    sns.lineplot(data=df, x='num_items_log2', y='throughput_per_pass_GBps', hue='Key', style='Key', markers=True, markersize=8, ax=ax1, legend=True)
    ax1.set_xticks(np.unique(df['num_items_log2']))
    ax1.set_ylabel('Throughput (GB/s)', fontsize=24)

    adjust_ax(
        ax1,
        fontsize=24,
        legend_fontsize=20,
        title=f"Sort Pair Performance",
        xlabel={"label": "Number of Items (log2 Scale)", "padding": 5},
        ylabel={"label": "Throughput (GB/s)"}
    )
    
    plt.tight_layout()
    save_fig(fig, f"sort_pair_throughput_{dict_to_str(params)}", subdir=f"archim/{machine}", suffix="pdf")

    fig, ax2 = create_plot((8,6))

    sns.barplot(data=df, x='num_items_log2', y='time_per_pass_msec', hue='Key', ax=ax2)
    ax2.set_ylabel('Time per Pass (ms)', fontsize=24)

    adjust_ax(
        ax2,
        fontsize=24,
        legend_fontsize=20,
        title=f"Sort Pair Performance",
        xlabel={"label": "Number of Items (log2 Scale)", "padding": 5},
        ylabel={"label": "Time per Pass (ms)"}
    )
    
    plt.tight_layout()
    save_fig(fig, f"sort_pair_time_{dict_to_str(params)}", subdir=f"archim/{machine}", suffix="pdf")

def plot_read_write_performance(csv_path, machine, **params):
    df = pd.read_csv(csv_path)
    df = df[df['GPU Time (sec)'].notna()]
    df = filter_df(
        df, 
        **params, 
        strict=True)
    df['throughput_GBps'] = (df['InputRead (bytes)'] + df['OutputWrite (bytes)']) / df['GPU Time (sec)'] / 1e9
    df['num_items_log2'] = np.log2(df['num_items'])

    fig, ax = create_plot((12,6))
    sns.lineplot(data=df, x='num_items_log2', y='throughput_GBps', hue='ValueType', style = 'ValueType', markers=True, markersize=8,ax=ax)
    ax.set_xticks(np.unique(df['num_items_log2'])[0::2])

    adjust_ax(
        ax,
        fontsize=24,
        legend_fontsize=20,
        title=f"Read/Write Performance",
        xlabel={"label": "Number of Items (log2 Scale)", "padding": 5},
        ylabel={"label": "Throughput (GB/s)"}
    )
    
    plt.tight_layout()
    save_fig(fig, f"read_write_{dict_to_str(params)}", subdir=f"archim/{machine}", suffix="pdf")

def plot_multiset(csv_path, machine, benchmark, **params):
    df = pd.read_csv(csv_path)
    df = df[df['GPU Time (sec)'].notna()]
    df = filter_df(
        df, 
        **params, 
        strict=True)
    
    df['build_num_keys'] = df['unique_keys'] * df['build_dup']
    df['build_num_keys_log2'] = np.log2(df['build_num_keys'])
    df['key_bytes'] = df['Key'].apply(lambda x: 4 if x == 'I32' else 8)
    df['input_bytes'] = df['key_bytes'] * df['build_num_keys']
    df['throughput_GBps'] = df['input_bytes'] / df['GPU Time (sec)'] / 1e9

    fig, ax = create_plot((12,6))
    sns.lineplot(data=df, x='build_num_keys_log2', y='throughput_GBps', hue='build_dup', style = 'CGSize', markers=True, markersize=8, ax=ax, palette='Set2')
    ax.set_xticks(np.unique(df['build_num_keys_log2']))
    ax.grid(True, which='both', axis='both', linestyle=':', alpha=0.75, zorder=0)
    adjust_ax(
        ax,
        fontsize=24,
        legend_fontsize=20,
        title=f"Multiset {benchmark} Performance",
        xlabel={"label": "Number of Items (log2 Scale)", "padding": 5},
        ylabel={"label": "Throughput (GB/s)"}
    )
    
    plt.tight_layout()
    save_fig(fig, f"multiset_{benchmark}_{dict_to_str(params)}", subdir=f"archim/{machine}", suffix="pdf")

if __name__ == "__main__":
    machines = ['gpu06']
    resdir = pathlib.Path(__file__).parent.absolute() / ".."  / ".." / "exp_results" / "archim"
    for machine in machines:
        for read_input in [1]:
            for reduce in [1]:
                for hashfn in [
                    'hash_xxhash_32', 
                    'hash_xxhash_64', 
                    'hash_murmurhash3_32', 
                    'hash_murmurhash3_fmix_32', 
                    'hash_murmurhash3_fmix_64']:
                    plot_hashing_performance(
                        latest_result_file(resdir / machine / f"cuco_hashfn_{machine}"), 
                        machine,
                        read_input=read_input, 
                        reduce=reduce, 
                        HashFunction=hashfn)
    
    for machine in machines:
        plot_gather_performance(latest_result_file(resdir / machine / f"gather_{machine}"), machine)
        plot_sort_pair_performance(latest_result_file(resdir / machine / f"sort_sweep_bits_{machine}"), machine)
        plot_read_write_performance(latest_result_file(resdir / machine / f"read_write_{machine}"), machine)
    
    for machine in machines:
        for key in ['I32', 'I64']:
            for load_factor in [0.25, 0.5]:
                for build_dup in [1, 2, 4]:
                    plot_multiset(
                        latest_result_file(resdir / machine / f"multiset_insert_{machine}"), 
                        machine,
                        "Insert",
                        Key=key,
                        load_factor=load_factor,
                        build_dup=build_dup)
    
        for machine in machines:
            for key in ['I32', 'I64']:
                for load_factor in [0.25, 0.5]:
                    for build_dup in [1, 2, 4]:
                        for hit_ratio in [1.0]:
                            plot_multiset(
                                latest_result_file(resdir / machine / f"multiset_count_build_sweep_{machine}"), 
                                machine,
                                "Count",
                                Key=key,
                                load_factor=load_factor,
                                build_dup=build_dup,
                                hit_ratio=hit_ratio)
        
        for machine in machines:
            for key in ['I32', 'I64']:
                for load_factor in [0.25, 0.5]:
                    for build_dup in [1, 2, 4]:
                        for hit_ratio in [1.0]:
                            plot_multiset(
                                latest_result_file(resdir / machine / f"multiset_probe_build_sweep_{machine}"), 
                                machine,
                                "Probe",
                                Key=key,
                                load_factor=load_factor,
                                build_dup=build_dup,
                                hit_ratio=hit_ratio)