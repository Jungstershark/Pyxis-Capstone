import cv2
import os
import shutil


def video_to_frames(video_path, output_dir, start_count=0, frames_per_second=10):
    """
    Extract frames from a video file and save them as sequentially numbered images.

    Parameters
    ----------
    video_path : str
        Path to the input .mp4 video file.

    output_dir : str
        Directory where extracted frames will be saved. The directory is created
        automatically if it does not exist.

    start_count : int, optional (default=0)
        The starting index used for naming the output frame files. This allows
        the function to participate in a global continuous numbering system when
        used in a larger dataset pipeline.

    frames_per_second : int, optional (default=10)
        Number of frames to extract per second of video. The function uses the
        video's native FPS to calculate sampling intervals.

    Behavior
    --------
    - Opens and reads the video frame-by-frame.
    - Calculates how frequently frames should be saved based on the target
      frames_per_second.
    - Saves frames as JPEG images using names:
          frame_00000.jpg, frame_00001.jpg, ...
    - Continues numbering from `start_count`, enabling global numbering
      across multiple videos or mixed data types.
    - Skips frames according to the calculated interval.
    - Automatically creates output_dir if not already present.

    Returns
    -------
    int
        The next available index after saving frames. This is used by the
        caller to maintain a global consecutive numbering scheme.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return start_count

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / frames_per_second) if video_fps > 0 else 1

    frame_count = 0
    counter = start_count
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{counter:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            counter += 1
        
        frame_count += 1
    
    cap.release()
    return counter


def process_dataset(input_folder, output_folder, frames_per_second=10):
    """
    Process a dataset of videos and images by extracting frames from videos and
    copying images into a single output directory using a unified naming scheme.

    Parameters
    ----------
    input_folder : str
        Root directory containing input videos and/or images. Nested folders
        are supported and will be scanned recursively.

    output_folder : str
        Destination directory where all processed files are saved. Will be
        created automatically if it does not exist.

    frames_per_second : int, optional (default=10)
        Extraction rate for .mp4 videos. Passed directly into video_to_frames().

    Behavior
    --------
    - Walks every subdirectory inside input_folder (recursive).
    - For each .mp4 video:
        * Calls video_to_frames() and extracts frames.
        * Uses and updates a global counter so every frame gets a unique,
          consecutive name.
    - For each image (.jpg/.jpeg/.png):
        * Copies the image into output_folder and assigns a name in the same
          global frame_X.jpg format used for video frames.
    - Ensures no filename collisions.
    - Produces a single, clean dataset where frames from all sources share one
      continuous numbering sequence.
    - Prints progress and a final summary.

    Returns
    -------
    None
        The function performs filesystem operations and prints status messages
        but does not return data.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_exts = {".jpg", ".jpeg", ".png"}

    global_counter = 0
    video_frames_saved = 0
    images_saved = 0

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            # VIDEO
            if ext == ".mp4":
                print(f"[VIDEO] Found: {file_path}")
                new_counter = video_to_frames(
                    file_path,
                    output_folder,
                    start_count=global_counter,
                    frames_per_second=frames_per_second
                )
                video_frames_saved += (new_counter - global_counter)
                global_counter = new_counter

            # IMAGE
            elif ext in image_exts:
                new_name = f"frame_{global_counter:05d}{ext}"
                dst = os.path.join(output_folder, new_name)

                shutil.copy(file_path, dst)
                images_saved += 1
                global_counter += 1

                print(f"[IMAGE] Copied: {file_path} -> {dst}")

    total = video_frames_saved + images_saved
    print(f"\n[DONE] All content saved to: {output_folder}")
    print(f"Video Frames: {video_frames_saved} | Images: {images_saved} | Total: {total}")


def delete_files_by_extension(dir_path, ext):
    """
    Delete all files in a directory that match one or more extensions.

    Parameters
    ----------
    dir_path : str
        Path to the directory whose files should be removed.

    ext : str or list/tuple of str
        The file extension(s) to delete. Examples:
        - ".jpg"
        - [".jpg", ".png"]
        - (".mp4", ".avi")

    Notes
    -----
    - Matching is case-insensitive (".JPG" will match ".jpg").
    - Only files in the top-level directory are removed (no recursion).
    - Extensions must include the leading dot, e.g., ".txt".
    """

    # Normalize ext to a tuple
    if isinstance(ext, str):
        exts = (ext.lower(),)
    else:
        exts = tuple(e.lower() for e in ext)

    for filename in os.listdir(dir_path):
        full_path = os.path.join(dir_path, filename)

        if os.path.isfile(full_path):
            if filename.lower().endswith(exts):
                os.remove(full_path)


def replace_spaces_in_filenames(dir_path, recursive=False):
    """
    Safely rename files so that spaces (" ") in filenames are replaced with
    underscores ("_"). Prevents overwriting by auto-generating unique names.

    Parameters
    ----------
    dir_path : str
        Path to the directory where filenames should be sanitized.

    recursive : bool, optional (default=False)
        If True, the function will rename files in all subdirectories as well.
        If False, only the top-level directory is processed.

    Behavior
    --------
    - Replaces all spaces in filenames with underscores.
    - Ensures no filename conflicts:
        If the new filename already exists, it appends a number:
        Example: "my file.txt" → "my_file.txt"
                 if "my_file.txt" exists → "my_file_1.txt", "my_file_2.txt", ...
    - Only renames files (not directories) unless recursive=True.
    """

    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            if " " in filename:
                old_path = os.path.join(root, filename)
                new_name = filename.replace(" ", "_")
                new_path = os.path.join(root, new_name)

                # If the sanitized name already exists, append numbers
                if os.path.exists(new_path):
                    base, ext = os.path.splitext(new_name)
                    counter = 1
                    while True:
                        candidate = f"{base}_{counter}{ext}"
                        new_path = os.path.join(root, candidate)
                        if not os.path.exists(new_path):
                            break
                        counter += 1

                print(f"Renaming: {old_path} → {new_path}")
                os.rename(old_path, new_path)

        if not recursive:
            break
