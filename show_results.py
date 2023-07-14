from PIL import Image, ImageDraw, ImageFont
import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("root_path")
args = parser.parse_args()


with open(os.path.join(args.root_path, "output.csv"), newline='') as csvfile:
    rowreader = csv.DictReader(csvfile)
    for row in rowreader:
        print(row)
        output_file = os.path.join(args.root_path, "labeled_output", row['deployment'], row['filename'])

        try:
            image = Image.open(output_file)
        except:
            image = Image.open(os.path.join(args.root_path, row['deployment'], row['filename']))


        draw = ImageDraw.Draw(image)
        
        top_left = (int(row['bbox_x']), int(row['bbox_y']))
        bottom_right = (top_left[0] + int(row['bbox_width']), top_left[1] + int(row['bbox_height']))
        draw.rectangle((top_left, bottom_right), outline="red", width=3)


        distance = round(float(row['animal_depth']), ndigits=1)
        text = f"{distance} m"
        font = ImageFont.truetype("arial.ttf", size=24)
        bbox = draw.textbbox(top_left, text, font=font)
        draw.rectangle(bbox, fill="black")
        draw.text(top_left, text, fill="white", font=font)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        image.save(output_file)