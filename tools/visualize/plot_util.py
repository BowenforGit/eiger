import pandas as pd
import os
import matplotlib
from matplotlib import pyplot as plt
import json
from collections.abc import Callable

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

which_cudf_stats = os.getenv("CUDF_RESULTS_STATS", "min")
print(f"Using CUDF_RESULTS_STATS={which_cudf_stats} for cuDF results")

def json_to_df(path_to_json: str, fn: Callable | None = None):
    if fn is None:
        fn = lambda x: x
    
    data = []
    with open(path_to_json, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            data.append(fn(json.loads(line)))
    
    return pd.DataFrame(data)

def filter_df(df, strict = False, must_have_attr=None, **filters):
    if must_have_attr:
        for attr in must_have_attr:
            if attr not in filters:
                raise ValueError(f"Attribute {attr} not found in specified filters but it's mandatory")
            
    for key, value in filters.items():
        if key not in df.columns:
            if strict:
                raise ValueError(f"Attribute {key} not found in dataframe")
            else:
                print(f"[WARNING] Attribute {key} not found in dataframe")
                continue
        
        if isinstance(value, list):
            df = df[df[key].isin(value)]
        elif callable(value):
            df = df[value(df[key])]
        else:
            df = df[df[key] == value]

        if len(df) == 0:
            break
        
    return df

def mark_ours(col, ours=[]):
    col = col.map(lambda x: f"{x} (Ours)" if x in ours else x)
    return col

def save_fig(fig, name, subdir="", suffix="jpg", dpi=300):
    resdir = os.path.join(os.path.dirname(__file__), os.pardir, "results", suffix, subdir)
    if not os.path.exists(resdir):
        os.makedirs(resdir)
    
    path = os.path.join(resdir, f"{name}.{suffix}")
    
    fig.tight_layout()
    
    if suffix == "jpg":
        fig.savefig(path, dpi=dpi)
    else:
        fig.savefig(path)
    
    print(f"Saved figure to {path}")

def dict_to_str(d):
    return "_".join([f"{key}_{value}" for key, value in d.items() if type(value) != list])

def create_plot(figsize, dim=(1, 1)):
    return plt.subplots(dim[0], dim[1], figsize=figsize)

def latest_result_file(prefix_path, suffix="csv"):
    dir_path = os.path.dirname(prefix_path)
    base_name = os.path.basename(prefix_path)
    candidates = []
    for file in os.listdir(dir_path):
        if file.startswith(base_name) and file.endswith(f".{suffix}"):
            candidates.append(file)
    
    if not candidates:
        raise FileNotFoundError(f"No files found with prefix {prefix_path}")
    
    candidates = sorted(candidates)
    latest_file = candidates[-1]

    print(f"Using latest result file: {latest_file}")
    
    return os.path.join(dir_path, latest_file)

def adjust_ax(ax, **params):
    if "fontsize" in params:
        fontsize = params["fontsize"]
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)

    if "legend_fontsize" in params:
        legend_fontsize = params["legend_fontsize"]
        if ax.get_legend() is not None:
            # set the legend font size without using ax.legend
            for legend_text in ax.get_legend().get_texts():
                legend_text.set_fontsize(legend_fontsize)
            
            if ax.get_legend().get_title() is not None:
                ax.get_legend().get_title().set_fontsize(legend_fontsize)
    
    if "title" in params:
        ax.set_title(params["title"])
    
    if "xlabel" in params:
        ax.set_xlabel(params["xlabel"].get("label", ax.get_xlabel()))
        ax.xaxis.labelpad = params["xlabel"].get("padding", 0)
    
    if "ylabel" in params:
        ax.set_ylabel(params["ylabel"].get("label", ax.get_ylabel()), rotation=params["ylabel"].get("rotation", 90))
        ax.yaxis.labelpad = params["ylabel"].get("padding", 0)
        if "x" in params["ylabel"] and "y" in params["ylabel"]:
            ax.yaxis.set_label_coords(params["ylabel"]["x"], params["ylabel"]["y"])
    
    if "xticklabels" in params:
        ax.set_xticklabels(params["xticklabels"].get("ticks", ax.get_xticklabels()), rotation=params["xticklabels"].get("rotation", 0))
    
    if "yticklabels" in params:
        ax.set_yticklabels(params["yticklabels"].get("ticks", ax.get_yticklabels()), rotation=params["yticklabels"].get("rotation", 0))
    
    if "xscale" in params:
        ax.set_xscale(params["xscale"].get("scale", ax.get_xscale()), base=params["xscale"].get("base", 10))
    
    if "yscale" in params:
        ax.set_yscale(params["yscale"].get("scale", ax.get_yscale()), base=params["yscale"].get("base", 10))
    
    if "ylim" in params:
        ax.set_ylim(params["ylim"])