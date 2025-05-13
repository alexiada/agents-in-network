import numpy as np
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def plot_network(position, node, site, exits, connections, ax, 
                 H_wall, LXmin, LXmax, LYmin, LYmax,):
    # Clear the axis for fresh plotting
    ax.clear()

    # Draw streets (connections) as rectangles aligned with the connection lines
    for start, end in connections:
        start_pos = node[start]
        end_pos = node[end]
        delta = end_pos - start_pos  # Vector for the connection
        length = np.linalg.norm(delta)  # Length of the connection

        # Compute the unit direction vector for the line
        unit_delta = delta / length

        # Compute the perpendicular vector for offsetting the rectangle
        perp_vector = np.array([-unit_delta[1], unit_delta[0]])

        # Rectangle corners (centred on the line)
        rectangle_corners = [
            start_pos + H_wall * perp_vector,
            start_pos - H_wall * perp_vector,
            end_pos - H_wall * perp_vector,
            end_pos + H_wall * perp_vector,
        ]

        # Create a polygon patch for the street
        street = patches.Polygon(rectangle_corners, closed=True, color='gray', alpha=0.5, zorder=1)
        ax.add_patch(street)

    # Draw nodes (excluding exits)
    for i, coord in enumerate(node):
        if i in exits:
            continue  # Skip plotting exits
        if i in site:
            # Draw sites as squares
            square = patches.Rectangle(
                coord - H_wall,  # Bottom-left corner
                2 * H_wall,      # Width
                2 * H_wall,      # Height
                color='blue',    # Colour for sites
                alpha=0.5,
                zorder=2
            )
            ax.add_patch(square)
        else:
            # Draw other nodes as circles
            circle = patches.Circle(coord, H_wall, color='gray', alpha=0.5, zorder=2)
            ax.add_patch(circle)

    # Plot particles as simple dots
    ax.scatter(position[:, 0], position[:, 1], color='red', s=1, label='Particles', zorder=5)

    # Annotate the nodes with their indices (excluding exits)
    for i, coord in enumerate(node):
        if i in exits:
            continue  # Skip annotating exits
        ax.text(coord[0] - 0.5 * H_wall, coord[1] + 0.5 * H_wall, f'{i}', fontsize=6, color='black', ha='center', zorder=3)

    # Configure plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal')  # Ensure equal scaling

    # Set fixed x and y axis limits
    ax.set_xlim(LXmin, LXmax)  # Replace with your desired x-axis limits
    ax.set_ylim(LYmin, LYmax)  # Replace with your desired y-axis limits

################################################################################################

def plot_network_with_incidents(incidents, node, site, exits, connections, ax,  
                                H_wall, LXmin, LXmax, LYmin, LYmax):
    # Clear the axis for fresh plotting
    ax.clear()

    # Draw streets (connections) as rectangles aligned with the connection lines
    for start, end in connections:
        start_pos = node[start]
        end_pos = node[end]
        delta = end_pos - start_pos  # Vector for the connection
        length = np.linalg.norm(delta)  # Length of the connection

        # Compute the unit direction vector for the line
        unit_delta = delta / length

        # Compute the perpendicular vector for offsetting the rectangle
        perp_vector = np.array([-unit_delta[1], unit_delta[0]])

        # Rectangle corners (centred on the line)
        rectangle_corners = [
            start_pos + H_wall * perp_vector,
            start_pos - H_wall * perp_vector,
            end_pos - H_wall * perp_vector,
            end_pos + H_wall * perp_vector,
        ]

        # Create a polygon patch for the street
        street = patches.Polygon(rectangle_corners, closed=True, color='gray', alpha=0.5, zorder=1)
        ax.add_patch(street)

    # Draw nodes (excluding exits)
    for i, coord in enumerate(node):
        if i in exits:
            continue  # Skip plotting exits
        if i in site:
            # Draw sites as squares
            square = patches.Rectangle(
                coord - H_wall,  # Bottom-left corner
                2 * H_wall,      # Width
                2 * H_wall,      # Height
                color='blue',    # Colour for sites
                alpha=0.5,
                zorder=2
            )
            ax.add_patch(square)
        else:
            # Draw other nodes as circles
            circle = patches.Circle(coord, H_wall, color='gray', alpha=0.5, zorder=2)
            ax.add_patch(circle)

    # Filter incidents to remove empty entries ([0,0,0])
    valid_incidents = incidents[~np.all(incidents == 0, axis=1)]  # Keep only valid incidents

    # Plot incidents as large red dots (alpha = 0.6 for transparency)
    if len(valid_incidents) > 0:
        ax.scatter(valid_incidents[:, 0], valid_incidents[:, 1], color='red', s=30, alpha=0.2, label='Incidents', zorder=5)

    # Annotate the nodes with their indices (excluding exits)
    for i, coord in enumerate(node):
        if i in exits:
            continue  # Skip annotating exits
        ax.text(coord[0] - 0.5 * H_wall, coord[1] + 0.5 * H_wall, f'{i}', fontsize=6, color='black', ha='center', zorder=3)

    # Configure plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal')  # Ensure equal scaling

    # Set fixed x and y axis limits
    ax.set_xlim(LXmin, LXmax)  # Replace with your desired x-axis limits
    ax.set_ylim(LYmin, LYmax)  # Replace with your desired y-axis limits

    ####################################################################################
import os
import shutil
import subprocess

def create_video(full_simulation, full_simulation_status, 
                 node, site, exits, connections, H_wall,
                 LXmin, LXmax, LYmin, LYmax, 
                 filename="simulation.mp4", frame_skip=1, highlighted_particles=[]):
    """
    frame_skip: Include only every `frame_skip` frame in the video.
    highlighted_particles: List of particles that should always be colored red.
    """
    # Ensure the frames directory exists
    os.makedirs("frames", exist_ok=True)

    # Generate frames
    print("Generating frames...")
    fig, ax = plt.subplots()

    # Select frames based on frame_skip
    selected_frames = full_simulation[::frame_skip]
    selected_states = full_simulation_status[::frame_skip]

    # Draw streets and nodes (excluding exits)
    for start, end in connections:
        start_pos = node[start]
        end_pos = node[end]
        delta = end_pos - start_pos
        length = np.linalg.norm(delta)

        unit_delta = delta / length
        perp_vector = np.array([-unit_delta[1], unit_delta[0]])

        rectangle_corners = [
            start_pos + H_wall * perp_vector,
            start_pos - H_wall * perp_vector,
            end_pos - H_wall * perp_vector,
            end_pos + H_wall * perp_vector,
        ]

        street = patches.Polygon(rectangle_corners, closed=True, color='gray', alpha=0.5, zorder=1)
        ax.add_patch(street)

    for i, coord in enumerate(node):
        if i in exits:
            continue  # Skip plotting exits
        if i in site:
            square = patches.Rectangle(
                coord - H_wall, 2 * H_wall, 2 * H_wall, color='blue', alpha=0.5, zorder=2
            )
            ax.add_patch(square)
        else:
            circle = patches.Circle(coord, H_wall, color='gray', alpha=0.5, zorder=2)
            ax.add_patch(circle)

    for i, coord in enumerate(node):
        if i in exits:
            continue  # Skip annotating exits
        ax.text(coord[0] - 0.5 * H_wall, coord[1] + 0.5 * H_wall, f'{i}', fontsize=6, color='black', ha='right', zorder=3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(LXmin, LXmax)
    ax.set_ylim(LYmin, LYmax)

    # Initialize particle scatter plot
    scatter = ax.scatter([], [], s=2, edgecolors='black', linewidths=0.1, marker='o', zorder=5) # Empty scatter plot for particles 

    for saved_frame_index, (position, status) in enumerate(zip(selected_frames, selected_states)):
        # Assign colors based on particle state
        color_map = {0: '#FFFF00', 1: '#00FF00', 2: '#FFFFFF', -1: '#FFFFFF'}  # Inside, Moving, Exiting, Out
        colors = [color_map[s] if i not in highlighted_particles else '#FF0000' for i, s in enumerate(status)]
 
        # Draw particles
        scatter.set_offsets(position)  # Update particle positions
        scatter.set_color(colors)  # Update particle colors
        scatter.set_edgecolors('black')  # Ensure the border is reapplied

        # Save the frame
        plt.savefig(f"frames/frame_{saved_frame_index:03d}.png", dpi=300)# defualt dpi=100

    plt.close(fig)  # Close the figure to free memory

    # Create video
    print("Creating video...")
    try:
        subprocess.run([
            "ffmpeg",
            "-y",                               # Overwrite output file without asking
            "-framerate", "30",                 # Set frame rate for the video
            "-i", "frames/frame_%03d.png",      # Input frame pattern
            "-c:v", "libx264",                  # Use H.264 codec
            "-preset", "slow",             # Quality encoding: ultrafast, fast, medium, slow, veryslow
            "-crf", "30",                       # Quality setting (lower is better) 30
            "-pix_fmt", "yuv420p",              # Ensure compatibility with most players
            filename                            # Output video file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("Video saved successfully as simulation.mp4")
    except subprocess.CalledProcessError as e:
        print("FFmpeg failed with the following error:")
        print(e.stderr.decode())

    # Cleanup frames directory
    print("Cleaning up frames...")
    shutil.rmtree("frames")  # Remove the frames directory after video creation
