import time
from collections import defaultdict
import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import shutil  # Added this import

# Start the timer
start_time = time.time()

# Load the YOLOv8 model to prep the program
model = YOLO('yolov8n.pt')
# load the actual model
model = YOLO('model.pt')

# Open the video file
video_path = "dustinjohnson.mp4" 
cap = cv2.VideoCapture(video_path)

# Get the total number of frames in the video for progress tracking
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Store the track history
track_history = defaultdict(lambda: [])

# Create 'frames' directory if it doesn't exist
if not os.path.exists("frames"):
    os.makedirs("frames")

frame_count = 0

# Initialize the tqdm progress bar
pbar = tqdm(total=total_frames, desc="Processing Frames", ncols=100)

# Loop through the video frames
while cap.isOpened():
    pbar.update(1)

    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        detected = False  # Use this flag to check if there are any detections in the current frame
        if hasattr(results[0].boxes, 'xywh') and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            detected = True
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))

        # Now draw the lines, even if there were no detections in this frame
        for track_id, track in track_history.items():
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=10)

        frame_name = os.path.join("frames", f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_name, frame)
        frame_count += 1

        if not detected:
            print(f"No detections in frame {frame_count}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Close the progress bar
pbar.close()

# After processing all frames, compile them into a video
frame_files = [os.path.join("frames", f) for f in os.listdir("frames") if f.startswith("frame_") and f.endswith(".png")]
frame_files.sort()

if frame_files:
    frame_example = cv2.imread(frame_files[0])
    height, width, layers = frame_example.shape
    size = (width, height)
    out = cv2.VideoWriter('compiled_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

    for frame_file in frame_files:
        img = cv2.imread(frame_file)
        out.write(img)

    out.release()

# Delete the 'frames' directory
shutil.rmtree("frames")

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# Calculate and display the elapsed time
elapsed_time = time.time() - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds")
