import os
import subprocess
import yaml
import re
import time
import tkinter as tk
from tkinter import filedialog
import shutil

def run_script():
    # Get variables from GUI entries
    SEQUENCE = sequence_entry.get()
    PATH_TO_VIDEO = path_to_video_entry.get()
    DOWNSAMPLE_RATE = int(downsample_rate_entry.get())
    SCENE_TYPE = scene_type_entry.get()
    EXPERIMENT = experiment_entry.get()
    GROUP = group_entry.get()
    NAME = name_entry.get()
    gpus = int(gpus_entry.get() or 1)
    max_iter = int(max_iter_entry.get())

    # Set variables as environment variables
    os.environ["SEQUENCE"] = SEQUENCE
    os.environ["PATH_TO_VIDEO"] = PATH_TO_VIDEO
    os.environ["DOWNSAMPLE_RATE"] = str(DOWNSAMPLE_RATE)
    os.environ["SCENE_TYPE"] = SCENE_TYPE
    os.environ["EXPERIMENT"] = EXPERIMENT
    os.environ["GROUP"] = GROUP
    os.environ["NAME"] = NAME
    os.environ["GPUS"] = str(gpus)

    config = f"projects/neuralangelo/configs/custom/{EXPERIMENT}.yaml"
    os.environ["CONFIG"] = config

    # Update the base.yaml file
    base_yaml_path = "projects/neuralangelo/configs/base.yaml"
    with open(base_yaml_path, "r") as file:
        base_yaml = yaml.safe_load(file)

    base_yaml["max_iter"] = max_iter

    with open(base_yaml_path, "w") as file:
        yaml.dump(base_yaml, file)

    # Run Git command
    print("Running Git command...")
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"])

    # Run preprocessing script
    print("Running preprocessing script...")
    subprocess.run(["bash", "projects/neuralangelo/scripts/preprocess.sh", SEQUENCE, PATH_TO_VIDEO, str(DOWNSAMPLE_RATE), SCENE_TYPE])

    # Function 1
    print("Running function 1...")
    process = subprocess.Popen(
        f"torchrun --nproc_per_node={os.environ['GPUS']} train.py "
        f"--logdir=logs/{os.environ['GROUP']}/{os.environ['NAME']} "
        f"--config={os.environ['CONFIG']} "
        f"--show_pbar",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    # Wait for the specific line and rename the checkpoint file
    checkpoint_pattern = re.compile(r"Saved checkpoint to (logs/.+/epoch_\d+_iteration_\d+_checkpoint\.pt)")

    for line in process.stdout:
        print(line, end="")
        match = checkpoint_pattern.search(line)
        if match:
            old_checkpoint = match.group(1)
            new_checkpoint = os.path.join(os.path.dirname(old_checkpoint), f"{os.environ['NAME']}.pt")
            os.rename(old_checkpoint, new_checkpoint)
            print(f"Renamed checkpoint file: {old_checkpoint} -> {new_checkpoint}")
            break

    process.wait()

    # Change directory permissions
    os.system("sudo chmod -R 777 logs")

    # Function 2
    print("Running function 2...")
    config = f"logs/{os.environ['GROUP']}/{os.environ['NAME']}/config.yaml"
    checkpoint = f"logs/{os.environ['GROUP']}/{os.environ['NAME']}/{os.environ['NAME']}.pt"
    output_mesh_folder = f"logs/{os.environ['GROUP']}/{os.environ['NAME']}/"
    output_mesh = f"logs/{os.environ['GROUP']}/{os.environ['NAME']}/{os.environ['NAME']}.ply"
    resolution = 2048
    block_res = 128
    os.system(f"torchrun --nproc_per_node={os.environ['GPUS']} projects/neuralangelo/scripts/extract_mesh.py "
              f"--config={config} "
              f"--checkpoint={checkpoint} "
              f"--output_file={output_mesh} "
              f"--resolution={resolution} "
              f"--block_res={block_res}")

    # Run f3d on the .ply file
    print("Running f3d to display the mesh...")
    subprocess.run(["f3d", output_mesh])

def browse_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    path_to_video_entry.delete(0, tk.END)
    path_to_video_entry.insert(tk.END, file_path)

# Create the main window
window = tk.Tk()
window.title("Neuralangelo Script")

# Create and position the labels and entry fields
sequence_label = tk.Label(window, text="Project Name:")
sequence_label.grid(row=0, column=0, sticky="e")
sequence_entry = tk.Entry(window)
sequence_entry.grid(row=0, column=1)

path_to_video_label = tk.Label(window, text="Path to Video:")
path_to_video_label.grid(row=1, column=0, sticky="e")
path_to_video_entry = tk.Entry(window)
path_to_video_entry.grid(row=1, column=1)
browse_button = tk.Button(window, text="Browse", command=browse_video)
browse_button.grid(row=1, column=2)

downsample_rate_label = tk.Label(window, text="Downsampling Rate:")
downsample_rate_label.grid(row=2, column=0, sticky="e")
downsample_rate_entry = tk.Entry(window)
downsample_rate_entry.grid(row=2, column=1)

scene_type_label = tk.Label(window, text="Scene Type:")
scene_type_label.grid(row=3, column=0, sticky="e")
scene_type_entry = tk.Entry(window)
scene_type_entry.grid(row=3, column=1)

experiment_label = tk.Label(window, text="Experiment Name:")
experiment_label.grid(row=4, column=0, sticky="e")
experiment_entry = tk.Entry(window)
experiment_entry.grid(row=4, column=1)

group_label = tk.Label(window, text="Group Name:")
group_label.grid(row=5, column=0, sticky="e")
group_entry = tk.Entry(window)
group_entry.grid(row=5, column=1)

name_label = tk.Label(window, text="Name:")
name_label.grid(row=6, column=0, sticky="e")
name_entry = tk.Entry(window)
name_entry.grid(row=6, column=1)

gpus_label = tk.Label(window, text="Number of GPUs:")
gpus_label.grid(row=7, column=0, sticky="e")
gpus_entry = tk.Entry(window)
gpus_entry.grid(row=7, column=1)

max_iter_label = tk.Label(window, text="Maximum Iterations:")
max_iter_label.grid(row=8, column=0, sticky="e")
max_iter_entry = tk.Entry(window)
max_iter_entry.grid(row=8, column=1)

# Create and position the run button
run_button = tk.Button(window, text="Run Script", command=run_script)
run_button.grid(row=9, column=0, columnspan=2, pady=10)

# Start the GUI event loop
window.mainloop()
