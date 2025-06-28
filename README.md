# Player-Re-Identification-in-a-Single-Video-Feed
This project detects and tracks players in a 15-second football video clip using a fine-tuned YOLOv11 model. The objective is to assign consistent player IDs, even when players leave and re-enter the frame.

📁 Folder Structure

ML_assignment/
├── main.ipynb
├── best.pt              # Provided YOLOv11 weights
├── 15sec_input_720p.mp4 # Input video
├── output.mp4           # (Generated) Output video with tracking
├── README.md


🚀 How to Run

1. Clone or Download

Place all the files including the video and best.pt model in the same folder.

2. Install Dependencies

pip install ultralytics opencv-python norfair

3. Run Jupyter Notebook

jupyter notebook

Open and execute main.ipynb step by step. This will:

Load the YOLOv11 model

Detect players in each frame

Track players across frames using Norfair

Save the result in output.mp4

link to YOLO weights
https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view

🧠 Components

YOLOv11 Detector (detector.py): Detects players in each frame.

Norfair Tracker (tracker.py): Maintains player identity using centroid-based tracking.

main.ipynb: Combines everything and handles the full video pipeline.

⚙️ Requirements

Python 3.8+

ultralytics (YOLO)

opencv-python

norfair

📌 Notes

Only class 0 (players) are tracked. Ball tracking can be added similarly.

Detection threshold can be tuned in detector.py.


Submitted as part of the Player Re-Identification Assignment (June 2025)

