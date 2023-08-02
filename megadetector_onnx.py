import onnxruntime
import numpy as np
import os
import cv2
import argparse


def read_as_numpy(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, axes=(2,0,1))
    return image


class MegaDetectorRuntime:
    def __init__(self):
        self.inference_session = onnxruntime.InferenceSession("./md_v5a.0.0.onnx")
        # onnx file downloaded from here: https://github.com/timmh/MegaDetectorLite/releases/tag/v0.2
    
    def run(self, filepath, conf_threshold=0.2):

        numpy_image = read_as_numpy(filepath)
        n_bands, image_height, image_width = numpy_image.shape
        md_output = self.inference_session.run(None, {"images": [numpy_image]})

        # iterate through detections and put them in the same format as returned by the megadetector API

        image_results = {"file": os.path.basename(filepath), "detections": []}

        for i in range(len(md_output[0])):
            # note: md_output has float32 numbers, convert to float64 with float() to play nice with json

            conf = float(md_output[0][i])
            if conf < conf_threshold:
                continue

            bbox_raw = md_output[2][i]
            bbox = [
                bbox_raw[0] / image_width,
                bbox_raw[1] / image_height,
                (bbox_raw[2] - bbox_raw[0]) / image_width,
                (bbox_raw[3] - bbox_raw[1]) / image_height
            ]
            bbox = list(map(lambda n: round(float(n), 4), bbox))

            image_results["detections"].append({
                "category": str(md_output[1][i] + 1),
                "conf": conf,
                "bbox": bbox
            })
        
        return image_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filepath")
    args = parser.parse_args()

    runtime = MegaDetectorRuntime()
    results = runtime.run(args.input_filepath)

    import json
    print(json.dumps(results, indent=1))