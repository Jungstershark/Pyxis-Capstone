from process_data import process_dataset, delete_files_by_extension, replace_spaces_in_filenames
import os

cur_dir = os.getcwd()
input_dir_1 = os.path.join(cur_dir, "..", "data", "raw_dataset", "pilot_transfer_pictures")
input_dir_2 = os.path.join(cur_dir, "..", "data", "raw_dataset", "pilot_transfer_videos")
output_dir = os.path.join(cur_dir,  "processed_data_22022026-1fps")

# test_video_dir = os.path.join(input_dir_2, "Dangerous_Pilot_Transfer_31.mp4")


if __name__ == "__main__":
    print(f"Input directory: {input_dir_1}, {input_dir_2} | Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    delete_files_by_extension(output_dir, [".jpg", ".png", ".jpeg"])
    replace_spaces_in_filenames(input_dir_1, True)
    replace_spaces_in_filenames(input_dir_2, True)

    last_counter = process_dataset(input_dir_1, output_dir, frames_per_second=1, return_counter=True)
    process_dataset(input_dir_2, output_dir, start_count=last_counter+1, frames_per_second=1)
