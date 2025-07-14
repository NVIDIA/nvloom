#!/usr/bin/env python3
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

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import sys
from collections.abc import Iterator


def get_arg_parser() -> argparse.ArgumentParser:
    usage_string = "pipe the output of nvloom_cli into plot_heatmaps.py\ncat output.txt | python3 plot_heatmaps.py"
    parser = argparse.ArgumentParser(prog="plot_heatmaps.py",
                                     usage=usage_string)
    parser.add_argument("-p", "--path", default=os.getcwd())
    parser.add_argument("-l", "--heatmap_lower_limit", default=0)
    parser.add_argument("-u", "--heatmap_upper_limit")
    parser.add_argument("--title_fontsize", default=32)
    parser.add_argument("--legend_fontsize", default=24)
    parser.add_argument("--data_fontsize_scaling_factor", default=1)
    parser.add_argument("--plot_size", default=32)
    parser.add_argument("--file_format", default="png", choices=list(plt.gcf().canvas.get_supported_filetypes().keys()))
    return parser


ARGS = get_arg_parser().parse_args()
TEXTCOLORS = ["black", "white"]


def convert_results_to_float(results: list[list[str]]) -> list[list[float]]:
    return [[float(x) if x != "N/A" else float("NAN") for x in y] for y in results]


def flatten_results(results: list[list[float]]) -> Iterator[float]:
    for x in results:
        for y in x:
            if not math.isnan(y):
                yield y


def generate_stats_plot(stats: plt.Axes,
                        additional_values: list[float],
                        color_threshold: float,
                        fontsize: float) -> None:
    im_stats = stats.imshow([additional_values], cmap="YlGn")
    im_stats.set_clim(ARGS.heatmap_lower_limit, ARGS.heatmap_upper_limit)
    for i in range(len(additional_values)):
        stats.text(i, 0, f"{additional_values[i]:.0f}",
                   ha="center",
                   va="center",
                   color=TEXTCOLORS[int(additional_values[i]) > color_threshold],
                   fontsize=fontsize)
    stats.set_xticks(np.arange(3), labels=["min", "avg", "max"], fontsize=ARGS.legend_fontsize)
    stats.set_xticks(np.arange(2)+.5, minor=True)
    stats.set_yticks([])
    stats.spines[:].set_visible(False)
    stats.set_title("summary", fontsize=ARGS.title_fontsize)
    stats.grid(which="minor", color="w", linestyle="-", linewidth=3)
    stats.tick_params(which="minor", bottom=False, left=False)


def apply_row_separators(heatmap: plt.Axes,
                         results: list[list[float]],
                         row_separators: list[int],
                         rack_guids: list[str]) -> None:
    if len(row_separators) == 0:
        return

    # we only want row separators if we have as many rows as processes
    if len(results) is not len(rack_guids):
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


def get_stats_values(results: list[list[float]]) -> tuple[float, float, float]:
    results_flattened = list(flatten_results(results))
    return min(results_flattened), sum(results_flattened)/len(results_flattened), max(results_flattened)


def validate_limits(min_value: float, max_value: float) -> None:
    if ARGS.heatmap_upper_limit:
        assert min_value < float(ARGS.heatmap_upper_limit), \
               f"heatmap_upper_limit: {float(ARGS.heatmap_upper_limit)} is smaller than minimum value " \
               f"of plotted results: {min_value}. Aborting"

    if ARGS.heatmap_lower_limit:
        assert max_value > float(ARGS.heatmap_lower_limit), \
               f"heatmap_lower_limit: {float(ARGS.heatmap_lower_limit)} is bigger than maximum value " \
               f"of plotted results: {max_value}. Aborting"


def get_data_fontsize(results: list[list[float]]) -> float:
    results_flattened = list(flatten_results(results))
    max_value_len = math.ceil(math.log10(max(results_flattened)))
    max_value_scaling_factor = math.pow((4. / max(max_value_len, 4)), 2.5)
    max_dim = max(len(results), len(results[0]))
    return ARGS.data_fontsize_scaling_factor * 60. * max_value_scaling_factor * (6. / max_dim)


def format_heatmap(heatmap: plt.Axes,
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
    heatmap.set_title(name, fontsize=ARGS.title_fontsize)
    main_heatmap_labelsize = 9 * math.pow(72 / len(results), 1/5)
    heatmap.tick_params(labelbottom=True, labeltop=True, labelsize=main_heatmap_labelsize)


def create_main_heatmap(heatmap: plt.Axes, results: list[list[float]]) -> None:
    im = heatmap.imshow(results, cmap="YlGn")
    # set heatmap color limits
    im.set_clim(ARGS.heatmap_lower_limit, ARGS.heatmap_upper_limit)

    cbar = heatmap.figure.colorbar(im, ax=heatmap)
    cbar.ax.tick_params(labelsize=ARGS.legend_fontsize)
    cbar.ax.set_ylabel("GB/s", rotation=-90, va="bottom", labelpad=20, fontsize=ARGS.legend_fontsize)


def plot_result(name: str,
                results: list[list[str]],
                labels_x: list[str],
                labels_y: list[str],
                columns_separators: list[int],
                row_separators: list[int],
                rack_guids: list[str],
                path: pathlib.Path) -> None:
    # printing testcase name to show progress in console output
    print(f"Plotting {name}")

    results = convert_results_to_float(results)

    # if values are NaN, there's nothing to plot
    if all(all(math.isnan(elem) for elem in sublist) for sublist in results):
        return

    min_value, avg_value, max_value = get_stats_values(results)
    validate_limits(min_value, max_value)

    fig, (heatmap, stats) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [math.sqrt(len(labels_y)), 1]})
    create_main_heatmap(heatmap, results)

    fontsize = get_data_fontsize(results)

    # depending on the field value, we will be adjusting text color to make it visible
    threshold = max_value/2.

    # generate the "stats" plot, with min/avg/max
    stats_fontsize = math.floor(math.sqrt(len(labels_y))) * fontsize
    generate_stats_plot(stats, [min_value, avg_value, max_value], threshold, stats_fontsize)

    # create numerical labels for each heatmap field
    apply_heatmap_data_labels(heatmap, results, threshold, fontsize)

    format_heatmap(heatmap, results, labels_x, labels_y, columns_separators, row_separators, rack_guids, name)

    fig.tight_layout()
    fig.set_size_inches(ARGS.plot_size, ARGS.plot_size)
    try:
        plt.savefig(str(path / f"{name}.{ARGS.file_format}"))
    except Exception as e:
        print(f"Error saving plot to file: {e}")
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


def main() -> None:
    # where to draw lines between rows to indicate different racks
    row_separators = []
    process_to_gpu_list = []
    rack_guid_list = []

    path = pathlib.Path(ARGS.path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")

    lines = sys.stdin.read().split("\n")
    lines = [x for x in lines if x != "" and not x.startswith("sleeping for")]

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
            while not lines[current_line].startswith("MinBW") and not len(lines[current_line].split()) == 0:
                result_values, label_y = parse_result_line(lines[current_line])
                results.append(result_values)
                labels_y.append(label_y)
                current_line += 1

            labels_y = enrich_labels_y(labels_y, process_to_gpu_list)

            try:
                plot_result(name,
                            results,
                            filtered_labels_x,
                            labels_y,
                            columns_separators,
                            row_separators,
                            rack_guid_list,
                            path)
            except Exception as e:
                print(f"Error plotting {name} heatmap: {e}")

        current_line += 1


if __name__ == "__main__":
    main()
