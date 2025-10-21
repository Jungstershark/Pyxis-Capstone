import cv2
import os

def video_to_frames(video_path, output_dir, frames_per_second=10):

    #create output directory if it dosent exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")

    #get video fps
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / frames_per_second)

    frame_count = 0
    saved_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        #save frame every "frame interval"
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames to f{output_dir}")
