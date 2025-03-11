#!/usr/bin/env python3

import os
import argparse
import matplotlib.pyplot as plt


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Image analysis",
        description="Extract and analyze/understand \
            the data set from the images and prompt \
            pie charts and bar charts for each \
            plant type."
    )
    parser.add_argument(
        "directory_name",
        help="The name of directory where the images are stored",
        type=str
    )
    return parser.parse_args()


def main(directory_name: str):
    if not os.path.isdir(directory_name):
        raise FileNotFoundError(
            f"{directory_name} directory not found or doesn't exists."
        )

    classes = {}
    for root, dirs, files in os.walk(directory_name):
        for file in files:
            plant_name = os.path.basename(root)
            plant_type = plant_name.split("_")[0]

            if plant_type not in classes:
                classes[plant_type] = {}
            if plant_type in classes and plant_name not in classes[plant_type]:
                classes[plant_type][plant_name] = []

            classes[plant_type][plant_name].append(file)

    fig, axs = plt.subplots(len(classes), 2, figsize=(11, 8))

    for ax, fruit in zip(axs, classes):
        for index, a in enumerate(ax):
            lengths = []
            labels = list(classes[fruit].keys())
            for _, images in classes[fruit].items():
                lengths.append(len(images))
            if index % 2 == 0:
                wedges, _, _ = \
                    a.pie(lengths, labels=labels, autopct='%1.1f%%')
                colors = {
                    label: wedge.get_facecolor()
                    for label, wedge in zip(labels, wedges)
                }
                a.set_title(f"{fruit} pie chart")
            else:
                bar_colors = [colors[label] for label in labels]
                a.bar(labels, lengths, color=bar_colors)
                a.set_xlabel("Plant names")
                a.set_ylabel("Number of images")
                a.set_title(f"{fruit} bar chart")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        args = args_parser()
        main(args.directory_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
