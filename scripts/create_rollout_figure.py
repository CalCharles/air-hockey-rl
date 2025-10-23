from PIL import Image, ImageSequence
import matplotlib.pyplot as plt

# List of GIF file paths (hardcoded for demonstration)
gif_filepaths = ["/home/mike/projects/air-hockey-rl/baseline_models/Puck V./air_hockey_agent_1/eval_0.gif", 
                 "/home/mike/projects/air-hockey-rl/baseline_models/Strike Crowd/air_hockey_agent_3/eval_0.gif", 
                 "/home/mike/projects/air-hockey-rl/baseline_models/Hit Goal/air_hockey_agent_1/eval_0.gif"] # Replace these paths with the actual paths to your GIFs

# Desired timesteps (in seconds) to extract frames
timesteps = [[11, 21, 29, 33],
             [1, 11, 17, 41],
             [11, 21, 60, 80]] # Replace these timesteps with the actual timesteps to extract

def extract_frames(gif_path, timesteps):
    """
    Extract frames from a GIF file at the specified timesteps.
    
    :param gif_path: Path to the GIF file.
    :param timesteps: List of timesteps in seconds to extract frames.
    :return: List of PIL Image objects of the extracted frames.
    """
    img = Image.open(gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
    # duration = img.info['duration'] / 1000.0 # Duration per frame in seconds
    
    # Calculate frame indices for the specified timesteps
    # indices = [int(t / duration) for t in timesteps]
    
    # Extract the frames
    extracted_frames = [frames[frame_idx] for frame_idx in timesteps]
    
    return extracted_frames

# Create a figure for the plots
fig, axs = plt.subplots(len(gif_filepaths), 4, figsize=(3*4, 4*4))

for i, gif_path in enumerate(gif_filepaths):
    frames = extract_frames(gif_path, timesteps[i])
    for j, frame in enumerate(frames):
        # Plot each frame in a subplot
        if len(gif_filepaths) == 1:
            ax = axs[j]
        else:
            ax = axs[i, j]
        ax.imshow(frame)
        ax.axis('off') # Hide axes for better visualization

# plt.tight_layout()
plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97, wspace=0.1, hspace=0.1)
plt.tight_layout(pad=0.1, w_pad=0.2, h_pad=2)
plt.savefig("box2d_rollout_example.pdf")
