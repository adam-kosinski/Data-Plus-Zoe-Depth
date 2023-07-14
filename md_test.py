import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

try:
    sys.path.append("MegaDetector")
    sys.path.append("yolov5")
    import run_detector_batch
    arg_string = f"megadetector_weights/md_v5a.0.0.pt {args.path} ./detections.json --threshold 0.2"
    run_detector_batch.main(arg_string)
except:
    import traceback
    traceback.print_exc()
finally:
    import time
    time.sleep(10)