from process_data import process_dataset, delete_files_by_extension, replace_spaces_in_filenames
import os

cur_dir = os.getcwd()
input_dir = os.path.join(cur_dir, "raw_dataset")
output_dir = os.path.join(cur_dir,  "processed_data")

test_video_dir = os.path.join(input_dir, "pilot_transfer_videos", "Dangerous_Pilot_Transfer_31.mp4")


if __name__ == "__main__":
    print(f"Input directory: {input_dir} | Output directory: {output_dir}")

    delete_files_by_extension(output_dir, [".jpg", ".png", ".jpeg"])
    replace_spaces_in_filenames(input_dir, True)

    process_dataset(input_dir, output_dir, frames_per_second=10)