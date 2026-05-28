import os
import glob
import imageio

def frames_to_video():
    image_folder = 'output_frames_shift'  
    video_name = 'falling_leaf_x_shift.mp4' 
    fps = 20                        
    
    search_path = os.path.join(image_folder, "frame_*.png")
    images = sorted(glob.glob(search_path))

    if not images:
        print(f"No images found! Please check if the '{image_folder}' directory is empty.")
        return

    print(f"Found {len(images)} image frames, generating MP4 (FPS={fps})...")
    

    writer = imageio.get_writer(video_name, fps=fps, macro_block_size=None)

    for img_path in images:
        img = imageio.v2.imread(img_path)
        writer.append_data(img)
        
    writer.close()
    print(f" Video generation complete! Saved as: {video_name}")

if __name__ == "__main__":
    frames_to_video()