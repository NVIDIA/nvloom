#!/usr/bin/env -S uv run --script
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ClusterShell~=1.0",
#     "matplotlib~=3.0",
#     "numpy~=2.0",
# ]
# ///

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import sys
import warnings
from collections.abc import Iterator
from ClusterShell.NodeSet import NodeSet


def get_arg_parser() -> argparse.ArgumentParser:
    usage_string = "pipe the output of nvloom_cli into plot_heatmaps.py\ncat output.txt | python3 plot_heatmaps.py"
    parser = argparse.ArgumentParser(prog="plot_heatmaps.py",
                                     usage=usage_string)
    parser.add_argument("-p", "--path", default=os.getcwd())
    parser.add_argument("-l", "--heatmap_lower_limit", default=0, type=float)
    parser.add_argument("-u", "--heatmap_upper_limit", type=float)
    parser.add_argument("--title_fontsize", default=32, type=float)
    parser.add_argument("--legend_fontsize", default=24, type=float)
    parser.add_argument("--data_fontsize_scaling_factor", default=1, type=float)
    parser.add_argument("--plot_size", default=32, type=float)
    parser.add_argument("--file_format", default="png", choices=list(plt.gcf().canvas.get_supported_filetypes().keys()))
    parser.add_argument("--no_heatmap_data_labels", action="store_true")
    parser.add_argument("--filename_prefix", default="")
    parser.add_argument("--histogram", default=False, action="store_true")
    parser.add_argument("--histogram_dynamic_min", default=False, action="store_true")
    parser.add_argument("--histogram_bins", default=20, type=int)
    return parser


TEXTCOLORS = ["black", "white"]


def convert_results_to_float(results: list[list[str]]) -> list[list[float]]:
    return [[float(x) if x != "N/A" else float("NAN") for x in y] for y in results]


def flatten_results(results: list[list[float]]) -> Iterator[float]:
    for x in results:
        for y in x:
            if not math.isnan(y):
                yield y


def generate_stats_plot(args: argparse.Namespace,
                        stats: plt.Axes,
                        additional_values: list[float],
                        color_threshold: float,
                        fontsize: float) -> None:
    im_stats = stats.imshow([additional_values], cmap="YlGn")
    im_stats.set_clim(args.heatmap_lower_limit, args.heatmap_upper_limit)
    for i in range(len(additional_values)):
        stats.text(i, 0, f"{additional_values[i]:.0f}",
                   ha="center",
                   va="center",
                   color=TEXTCOLORS[int(additional_values[i]) > color_threshold],
                   fontsize=fontsize)
    stats.set_xticks(np.arange(4), labels=["min", "avg", "med", "max"], fontsize=args.legend_fontsize)
    stats.set_xticks(np.arange(3)+.5, minor=True)
    stats.set_yticks([])
    stats.spines[:].set_visible(False)
    stats.set_title("summary", fontsize=args.title_fontsize)
    stats.grid(which="minor", color="w", linestyle="-", linewidth=3)
    stats.tick_params(which="minor", bottom=False, left=False)


def apply_row_separators(heatmap: plt.Axes,
                         results: list[list[float]],
                         row_separators: list[int],
                         rack_guids: list[str]) -> None:
    if len(row_separators) == 0:
        return

    # we only want row separators if we have as many rows as processes
    if len(results) != len(rack_guids):
        return

    sec = heatmap.secondary_yaxis(location="right")
    ids = [0] + row_separators
    additional_labels = [rack_guids[i] for i in ids]

    boundaries = ids + [len(results)]
    locations = []
    for i in range(len(boundaries) - 1):
        locations.append((boundaries[i] + boundaries[i+1])/2)

    sec.set_yticks(np.array(locations) - 0.5, labels=additional_labels)
    sec.tick_params('y', length=0)
    heatmap.hlines(y=[y - 0.5 for y in row_separators],
                   xmin=-0.5,
                   xmax=len(results[0]) - 0.5,
                   color="black",
                   linewidth=4)


def apply_column_separators(heatmap: plt.Axes,
                            results: list[list[float]],
                            columns_separators: list[int],
                            labels_x: list[str],
                            labels_y: list[str]) -> None:
    boundaries = [0] + columns_separators + [len(results[0])]
    locations = []
    for i in range(len(boundaries) - 1):
        locations.append((boundaries[i] + boundaries[i+1])/2 + 0.01)
    heatmap.set_xticks(np.array(locations), labels=labels_x)
    heatmap.tick_params(axis="x", which="both", length=0)
    add_vertical_lines(heatmap, columns_separators, labels_y)


def add_vertical_lines(heatmap: plt.Axes, columns_separators: list[int], labels_y: list[str]) -> None:
    heatmap.vlines(x=[x - 0.5 for x in columns_separators],
                   ymin=-0.5,
                   ymax=len(labels_y) - 0.5,
                   color="black",
                   linewidth=4)


def apply_heatmap_data_labels(heatmap: plt.Axes, results: list[list[float]], threshold: float, fontsize: float) -> None:
    for i in range(len(results)):
        for j in range(len(results[i])):
            if not math.isnan(results[i][j]):
                heatmap.text(j, i, f"{results[i][j]:.0f}",
                             ha="center",
                             va="center",
                             color=TEXTCOLORS[int(results[i][j]) > threshold],
                             fontsize=fontsize)


def get_stats_values(results: list[list[float]]) -> tuple[float, float, float, float]:
    results_flattened = sorted(list(flatten_results(results)))
    n = len(results_flattened)
    median = (results_flattened[(n - 1) // 2] + results_flattened[n // 2]) / 2
    return min(results_flattened), sum(results_flattened) / n, median, max(results_flattened)


def validate_limits(args: argparse.Namespace, min_value: float, max_value: float) -> None:
    if args.heatmap_upper_limit:
        assert min_value < float(args.heatmap_upper_limit), \
               f"heatmap_upper_limit: {float(args.heatmap_upper_limit)} is smaller than minimum value " \
               f"of plotted results: {min_value}. Aborting"

    if args.heatmap_lower_limit:
        assert max_value > float(args.heatmap_lower_limit), \
               f"heatmap_lower_limit: {float(args.heatmap_lower_limit)} is bigger than maximum value " \
               f"of plotted results: {max_value}. Aborting"


def get_data_fontsize(args: argparse.Namespace, results: list[list[float]]) -> float:
    results_flattened = list(flatten_results(results))
    max_value = max(results_flattened) if results_flattened else 1.0
    max_value_len = math.ceil(math.log10(max(max_value, 1.0)))
    max_value_scaling_factor = math.pow((4. / max(max_value_len, 4)), 2.5)
    max_dim = max(len(results), len(results[0]))
    return args.data_fontsize_scaling_factor * 60. * max_value_scaling_factor * (6. / max_dim)


def format_heatmap(args: argparse.Namespace,
                   heatmap: plt.Axes,
                   results: list[list[float]],
                   labels_x: list[str],
                   labels_y: list[str],
                   columns_separators: list[int],
                   row_separators: list[int],
                   rack_guids: list[str],
                   name: str) -> None:
    heatmap.spines[:].set_visible(False)

    heatmap.set_yticks(np.arange(len(results)-1)+.5, minor=True)
    heatmap.set_xticks(np.arange(len(results[0])-1)+.5, minor=True)
    if len(columns_separators) == 0 and len(labels_x) == len(results[0]):
        if len(rack_guids) == len(labels_x):
            add_vertical_lines(heatmap, row_separators, labels_y)
        heatmap.set_xticks(np.arange(len(labels_x)), labels=labels_x)
    else:
        apply_column_separators(heatmap, results, columns_separators, labels_x, labels_y)

    heatmap.set_yticks(np.arange(len(labels_y)), labels=labels_y)

    apply_row_separators(heatmap, results, row_separators, rack_guids)

    heatmap.grid(which="minor", color="w", linestyle="-", linewidth=3)
    heatmap.set_title(f"{name}\n{get_nodes_description(labels_y)}", fontsize=args.title_fontsize)
    main_heatmap_labelsize = 9 * math.pow(72 / len(results), 1/5)
    heatmap.tick_params(labelbottom=True, labeltop=True, labelsize=main_heatmap_labelsize)


def create_main_heatmap(args: argparse.Namespace,
                        heatmap: plt.Axes,
                        results: list[list[float]],
                        unit: str) -> None:
    im = heatmap.imshow(results, cmap="YlGn")
    # set heatmap color limits
    im.set_clim(args.heatmap_lower_limit, args.heatmap_upper_limit)

    cbar = heatmap.figure.colorbar(im, ax=heatmap)
    cbar.ax.tick_params(labelsize=args.legend_fontsize)
    cbar.ax.set_ylabel(unit, rotation=-90, va="bottom", labelpad=20, fontsize=args.legend_fontsize)


def plot_heatmap(args: argparse.Namespace,
                 name: str,
                 results: list[list[float]],
                 labels_x: list[str],
                 labels_y: list[str],
                 columns_separators: list[int],
                 row_separators: list[int],
                 rack_guids: list[str],
                 unit: str) -> plt.Figure:
    results = convert_results_to_float(results)

    # if values are NaN, there's nothing to plot
    if all(all(math.isnan(elem) for elem in sublist) for sublist in results):
        return

    min_value, avg_value, med_value, max_value = get_stats_values(results)
    validate_limits(args, min_value, max_value)

    fig, (heatmap, stats) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [math.sqrt(len(labels_y)), 1]})
    create_main_heatmap(args, heatmap, results, unit)

    fontsize = get_data_fontsize(args, results)

    # depending on the field value, we will be adjusting text color to make it visible
    threshold = max_value/2.

    # generate the "stats" plot, with min/avg/med/max
    stats_fontsize = math.floor(math.sqrt(len(labels_y))) * fontsize
    generate_stats_plot(args, stats, [min_value, avg_value, med_value, max_value], threshold, stats_fontsize)

    # create numerical labels for each heatmap field
    if not args.no_heatmap_data_labels:
        apply_heatmap_data_labels(heatmap, results, threshold, fontsize)

    format_heatmap(args, heatmap, results, labels_x, labels_y, columns_separators, row_separators, rack_guids, name)
    return fig


def get_unique_nodes_from_labels(nodes: list[str]) -> list[str]:
    nodes = [node.split("/")[0].split("->")[0] for node in nodes]
    return list(set(nodes))


def get_nodes_string(nodes: list[str]) -> str:
    return str(NodeSet(",".join(nodes)))


def get_nodes_description(nodes: list[str]) -> str:
    unique_nodes = get_unique_nodes_from_labels(nodes)
    nodes_string = get_nodes_string(unique_nodes)
    nodes_count = len(unique_nodes)
    return f"{nodes_string} ({nodes_count} nodes)" if nodes_count > 1 else nodes_string


def plot_histogram(args: argparse.Namespace,
                   name: str,
                   results: list[float],
                   nodes: list[str],
                   unit: str) -> plt.Figure:
    fig, ax = plt.subplots()
    fig.set_size_inches(args.plot_size, args.plot_size)
    if args.histogram_dynamic_min:
        range = (min(results), max(results))
    else:
        range = (0, max(results))
    ax.hist(list(results), bins=args.histogram_bins, color="green", edgecolor="white", range=range)
    ax.set_title(f"{name}\n{len(results)} total datapoints\n{get_nodes_description(nodes)}",
                 fontsize=args.title_fontsize)
    ax.set_xlabel(unit, fontsize=args.legend_fontsize)
    ax.set_ylabel("Count", fontsize=args.legend_fontsize)
    ax.grid(True)
    ax.tick_params(labelsize=args.legend_fontsize)
    fig.tight_layout()
    return fig


def plot_result(args: argparse.Namespace,
                name: str,
                results: list[list[str]],
                labels_x: list[str],
                labels_y: list[str],
                columns_separators: list[int],
                row_separators: list[int],
                rack_guids: list[str],
                path: pathlib.Path,
                unit: str) -> None:
    # printing testcase name to show progress in console output
    print(f"Plotting {name}")

    results = convert_results_to_float(results)

    # if values are NaN, there's nothing to plot
    if all(all(math.isnan(elem) for elem in sublist) for sublist in results):
        return

    if not args.histogram:
        fig = plot_heatmap(args, name, results, labels_x, labels_y, columns_separators, row_separators, rack_guids, unit)
    else:
        fig = plot_histogram(args, name, list(flatten_results(results)), labels_y, unit)

    # fig.tight_layout()
    fig.set_size_inches(args.plot_size, args.plot_size)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            plt.savefig(str(path / f"{args.filename_prefix}{name}.{args.file_format}"))
    except Exception as e:
        print(f"Error saving plot to file: {e}")
        plt.close(fig)
        raise
    plt.close(fig)


def prepare_row_separator_list(rack_guid_list: list[str]) -> list[int]:
    row_separators = []
    for idx, guid in enumerate(rack_guid_list):
        if idx + 1 < len(rack_guid_list):
            if rack_guid_list[idx + 1] != guid:
                row_separators.append(idx + 1)
    return row_separators


def parse_column_separators(labels_x: list[str]) -> tuple[list[str], list[int]]:
    have_dash = False
    counter = 0
    filtered_labels_x = []
    columns_separators = []
    for labelX in labels_x:
        if labelX == "|":
            columns_separators.append(counter)
        else:
            if "-" in labelX:
                have_dash = True
                if len(columns_separators) == len(filtered_labels_x):
                    filtered_labels_x.append(labelX.split("-")[0])
            else:
                filtered_labels_x.append(labelX)

            counter += 1
    if not have_dash:
        filtered_labels_x = labels_x
    return filtered_labels_x, columns_separators


def parse_result_line(line: str) -> tuple[list[str], str]:
    label_y = line.split()[0]
    result_values = line.split()[1:]
    result_values = [word for word in result_values if word != "|"]
    result_values = [word[0:word.find(";")] if ";" in word else word for word in result_values]
    return result_values, label_y


def parse_process_line(line: str) -> tuple[str, str]:
    words = line.split()
    hostname = words[2][1:-2].split(".")[0]
    GPU_id = words[4][:-1]
    rack_guid = words[-1]
    process_name = f"{hostname}/{GPU_id}"
    return process_name, rack_guid


# if labels_y is list of consecutive integers, replace it with process names
def enrich_labels_y(labels_y: list[str], process_to_gpu_list: list[str]) -> list[str]:
    default_list = list(map(str, list(range(len(labels_y)))))
    if default_list == labels_y and len(labels_y) == len(process_to_gpu_list):
        labels_y = process_to_gpu_list
    return labels_y


def parse_unit(line: str) -> str:
    if len(line.split()) < 4:
        return "GB/s"
    return line.split()[3]


def main() -> None:
    parser = get_arg_parser()

    if len(sys.argv) == 1 and sys.stdin.isatty():
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # where to draw lines between rows to indicate different racks
    row_separators = []
    process_to_gpu_list = []
    rack_guid_list = []

    path = pathlib.Path(args.path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")

    lines = sys.stdin.read().split("\n")
    lines = [x for x in lines if x != "" and not x.startswith("sleeping for")]

    failed = False

    current_line = 0
    while current_line < len(lines):
        # get information about processes
        if lines[current_line].startswith("Process"):
            process_name, rack_guid = parse_process_line(lines[current_line])
            process_to_gpu_list.append(process_name)
            rack_guid_list.append(rack_guid)

        # parse a testcase output
        if lines[current_line].startswith("Running") and len(lines[current_line].split()) == 2:

            if len(row_separators) == 0:
                row_separators = prepare_row_separator_list(rack_guid_list)

            name = lines[current_line].split()[1]
            current_line += 1

            if "WAIVED" in lines[current_line]:
                continue

            filtered_labels_x, columns_separators = parse_column_separators(lines[current_line].split())
            current_line += 1

            labels_y = []
            results = []
            while not lines[current_line].startswith("Min") and not len(lines[current_line].split()) == 0:
                result_values, label_y = parse_result_line(lines[current_line])
                results.append(result_values)
                labels_y.append(label_y)
                current_line += 1

            while not lines[current_line].startswith("Min"):
                current_line += 1

            unit = parse_unit(lines[current_line])

            labels_y = enrich_labels_y(labels_y, process_to_gpu_list)

            try:
                plot_result(args,
                            name,
                            results,
                            filtered_labels_x,
                            labels_y,
                            columns_separators,
                            row_separators,
                            rack_guid_list,
                            path,
                            unit)
            except Exception as e:
                print(f"Error plotting {name} heatmap: {e}")
                failed = True

        current_line += 1

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
