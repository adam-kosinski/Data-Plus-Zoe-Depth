import sys
import run_detector_batch

sys.path.append("MegaDetector")
sys.path.append("yolov5")
arg_string = "megadetector_weights/md_v5a.0.0.pt [file_dir] [output.json] --recursive --threshold 0.2"
run_detector_batch.main(arg_string)

arg_string = "megadetector_weights/md_v5a.0.0.pt test/first_cropped test/detections.json --recursive --threshold 0.2"
