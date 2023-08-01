from PIL import Image, ImageDraw, ImageFont
import csv
import os
import argparse
import numpy as np
from zoedepth.utils.misc import colorize


def label_results(root_path, csv_filepath="output.csv", dest_folder="labeled_output"):
    # file and dest folder are relative to the root path

    seen_rgb_already = []
    seen_depth_already = []

    with open(os.path.join(root_path, csv_filepath), newline='') as csvfile:
        rowreader = csv.DictReader(csvfile)
        for row in rowreader:
            
            # make dest directory if necessary
            os.makedirs(os.path.join(root_path, dest_folder, row['deployment']), exist_ok=True)


            # label RGB image
            rgb_output_file = os.path.join(root_path, dest_folder, row['deployment'], row['filename'])
            print(rgb_output_file)

            rgb_file_to_open = rgb_output_file if rgb_output_file in seen_rgb_already else os.path.join(root_path, "deployments", row['deployment'], row['filename'])
            with Image.open(rgb_file_to_open) as image:
                draw_annotations(image, row)
                image.save(rgb_output_file)
                seen_rgb_already.append(rgb_output_file)


            # label depth image
            name, ext = os.path.splitext(row['filename'])
            depth_output_file = os.path.join(root_path, dest_folder, row['deployment'], name + "_depth" + ext)

            if depth_output_file not in seen_depth_already:
                raw_depth_file = os.path.join(root_path, "depth_maps", row['deployment'], os.path.splitext(row['filename'])[0] + "_raw.png")
                with Image.open(raw_depth_file) as depth_img:
                    depth = np.asarray(depth_img) / 256
                    colored = colorize(depth)
                    Image.fromarray(colored).convert("RGB").save(depth_output_file)
            
            with Image.open(depth_output_file) as image:
                draw_annotations(image, row)
                image.save(depth_output_file)
                seen_depth_already.append(depth_output_file)


def draw_annotations(image, row):
    draw = ImageDraw.Draw(image)
            
    top_left = (int(row['bbox_x']), int(row['bbox_y']))
    bottom_right = (top_left[0] + int(row['bbox_width']), top_left[1] + int(row['bbox_height']))
    draw.rectangle((top_left, bottom_right), outline="red", width=3)

    radius = 10
    sample_top_left = (int(row['sample_x'])-radius, int(row['sample_y'])-radius)
    sample_bottom_right = (int(row['sample_x'])+radius, int(row['sample_y'])+radius)
    draw.arc((sample_top_left, sample_bottom_right), 0, 360, fill="red", width=5)

    distance = round(float(row['animal_distance']), ndigits=1)
    text = f"{distance} m"
    font = ImageFont.truetype("arial.ttf", size=24)
    bbox = draw.textbbox(top_left, text, font=font)
    draw.rectangle(bbox, fill="black")
    draw.text(top_left, text, fill="white", font=font)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path")
    args = parser.parse_args()
    label_results(args.root_path)