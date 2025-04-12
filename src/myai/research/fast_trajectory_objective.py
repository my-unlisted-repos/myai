import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional
from ..video import render_frames

# Helper function (remains the same)
def normalize_coords(coords: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
    """Normalizes coordinates from grid space to [-1, 1] for grid_sample."""
    H, W = grid_size
    coords_normalized_x = (coords[:, 0] / (W - 1)) * 2 - 1
    coords_normalized_y = (coords[:, 1] / (H - 1)) * 2 - 1
    normalized_coords = torch.stack([coords_normalized_x, coords_normalized_y], dim=1)
    return normalized_coords.unsqueeze(0).unsqueeze(0) # -> (1, 1, N, 2)

class MazeTrajectoryOptimization(nn.Module):
    """
    PyTorch nn.Module for optimizing a trajectory through a 2D maze.

    Optimizes a sequence of points to find a short, collision-free path
    from a start to an end point within a given maze. Penalizes collisions
    along path segments by sub-sampling.

    Args:
        maze (np.ndarray): A 2D numpy array representing the maze.
                           1 indicates free space, 0 indicates a wall.
        start_point (Tuple[float, float]): (x, y) coordinates of the start.
        end_point (Tuple[float, float]): (x, y) coordinates of the end.
        n_points (int): Number of key points in the trajectory (including start/end).
        n_subsample_points (int): Number of points to sample between each pair of
                                  key points for collision checking. 0 means only
                                  key points are checked.
        lambda_collision (float): Weight for the collision loss term.
        lambda_length (float): Weight for the trajectory length loss term.
        init_straight (bool): If True, initialize trajectory as a straight line.
                              Otherwise, use random initialization.
        optimize_start_end (bool): If False (default), start and end points are fixed.
                                   If True, they are included in optimization.
        device (Optional[str]): Device to run computations on ('cpu' or 'cuda').
    """
    def __init__(
        self,
        maze: np.ndarray,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        n_points: int = 20,
        n_subsample_points: int = 3, # <--- New parameter
        lambda_collision: float = 100.0,
        lambda_length: float = 1.0,
        init_straight: bool = True,
        optimize_start_end: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()

        if n_points < 2:
            raise ValueError("n_points must be at least 2 (start and end).")
        if n_subsample_points < 0:
             raise ValueError("n_subsample_points cannot be negative.")

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.maze = torch.from_numpy(maze.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device) # Shape (1, 1, H, W)
        self.grid_size = (maze.shape[0], maze.shape[1]) # H, W
        self.start_point = torch.tensor(start_point, dtype=torch.float32, device=self.device) # (x, y)
        self.end_point = torch.tensor(end_point, dtype=torch.float32, device=self.device)   # (x, y)
        self.n_points = n_points
        self.n_subsample_points = n_subsample_points # Store sub-sampling count
        self.lambda_collision = lambda_collision
        self.lambda_length = lambda_length
        self.optimize_start_end = optimize_start_end

        # Initialize Trajectory Points (same as before)
        if init_straight:
            t = torch.linspace(0, 1, n_points, device=self.device).unsqueeze(1)
            init_traj = self.start_point * (1 - t) + self.end_point * t
        else:
            t = torch.linspace(0, 1, n_points, device=self.device).unsqueeze(1)
            init_traj = self.start_point * (1 - t) + self.end_point * t
            noise_scale = min(self.grid_size) * 0.1
            init_traj[1:-1] += torch.randn(n_points - 2, 2, device=self.device) * noise_scale

        # Define Optimizable Parameters (same as before)
        if self.optimize_start_end:
            self.trajectory_points = nn.Parameter(init_traj)
        else:
            self.fixed_start = init_traj[0:1]
            self.fixed_end = init_traj[-1:]
            self.intermediate_points = nn.Parameter(init_traj[1:-1])

        # Visualization (same setup, using the corrected _create_base_maze_image)
        self.frames: List[np.ndarray] = []
        self._create_base_maze_image() # Use the corrected version here

    # Use the corrected version from the previous response
    def _create_base_maze_image(self):
        """Creates the background image for visualization."""
        H, W = self.grid_size
        scale = 10
        vis_img = np.zeros((H * scale, W * scale, 3), dtype=np.uint8)
        maze_np = self.maze.squeeze().cpu().numpy()
        free_y_indices, free_x_indices = np.where(maze_np == 1)
        for y, x in zip(free_y_indices, free_x_indices):
            y_start, y_end = y * scale, (y + 1) * scale
            x_start, x_end = x * scale, (x + 1) * scale
            vis_img[y_start:y_end, x_start:x_end, :] = 255 # White
        start_px = (int(round(self.start_point[0].item() * scale)), int(round(self.start_point[1].item() * scale)))
        end_px = (int(round(self.end_point[0].item() * scale)), int(round(self.end_point[1].item() * scale)))
        radius = max(1, scale // 2 - 1)
        cv2.circle(vis_img, start_px, radius, (0, 255, 0), -1) # Green
        cv2.circle(vis_img, end_px, radius, (0, 0, 255), -1)   # Red
        self.base_vis_img = vis_img
        self.vis_scale = scale

    def _get_current_trajectory(self) -> torch.Tensor:
        """Helper to get the full trajectory tensor."""
        if self.optimize_start_end:
            return self.trajectory_points
        else:
            return torch.cat([self.fixed_start, self.intermediate_points, self.fixed_end], dim=0)

    # Visualization drawing (remains the same)
    def _draw_trajectory(self, trajectory_np: np.ndarray) -> np.ndarray:
        """Draws the current trajectory onto a copy of the base maze image."""
        vis_img = self.base_vis_img.copy()
        scale = self.vis_scale
        traj_pixels = (trajectory_np * scale).astype(np.int32)
        for i in range(len(traj_pixels) - 1):
            pt1 = tuple(traj_pixels[i])
            pt2 = tuple(traj_pixels[i+1])
            cv2.line(vis_img, pt1, pt2, (255, 0, 0), max(1, scale // 5))
            cv2.circle(vis_img, pt1, max(1, scale // 4), (0, 255, 0), -1) # Green
        return vis_img


    def forward(self) -> torch.Tensor:
        """
        Calculates the trajectory optimization loss and generates visualization.
        Includes collision checks for sub-sampled points along segments.

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # --- Get Current Key Trajectory Points ---
        current_trajectory = self._get_current_trajectory() # Shape (n_points, 2) [x, y]

        # --- Generate Points for Collision Checking (Key points + Sub-sampled points) ---
        points_to_check = current_trajectory # Start with the key points

        if self.n_subsample_points > 0:
            # Generate interpolation factors (e.g., for n=3 -> 0.25, 0.5, 0.75)
            alphas = torch.linspace(0.0, 1.0, self.n_subsample_points + 2, device=self.device)[1:-1] # Exclude 0.0 and 1.0
            alphas = alphas.unsqueeze(0).unsqueeze(-1) # Shape (1, n_subsample, 1) for broadcasting

            # Get segments start and end points
            starts = current_trajectory[:-1].unsqueeze(1) # Shape (n_points-1, 1, 2)
            ends = current_trajectory[1:].unsqueeze(1)   # Shape (n_points-1, 1, 2)

            # Linear interpolation: p = start * (1-alpha) + end * alpha
            sub_sampled_points = starts * (1.0 - alphas) + ends * alphas # Shape (n_points-1, n_subsample, 2)

            # Reshape and concatenate with original points
            sub_sampled_points = sub_sampled_points.view(-1, 2) # Shape ((n_points-1)*n_subsample, 2)
            points_to_check = torch.cat([current_trajectory, sub_sampled_points], dim=0)


        # --- 1. Collision Loss (applied to all points_to_check) ---
        normalized_coords_to_check = normalize_coords(points_to_check, self.grid_size) # Shape (1, 1, N_total, 2)

        sampled_maze_values = F.grid_sample(
            self.maze,
            normalized_coords_to_check,
            mode='bilinear',
            padding_mode='border', # Important: Penalize going outside bounds too
            align_corners=True
        ) # Output shape (1, 1, 1, N_total)

        sampled_maze_values = sampled_maze_values.squeeze() # Shape (N_total,)
        # Penalize low values (walls=0) more. Clamp ensures non-negative loss.
        collision_penalty = (1.0 - sampled_maze_values).clamp(min=0.0) ** 2
        loss_collision = collision_penalty.mean() # Average penalty over ALL checked points

        # --- 2. Trajectory Length Loss (based on original key points) ---
        # Length loss should still be based on the main control points
        segments = current_trajectory[1:] - current_trajectory[:-1]
        segment_lengths = torch.sqrt(torch.sum(segments**2, dim=1))
        loss_length = torch.sum(segment_lengths)

        # --- 3. Start/End Point Constraint Loss (only if optimize_start_end is True) ---
        loss_start_end = torch.tensor(0.0, device=self.device)
        if self.optimize_start_end:
            start_diff = torch.sum((current_trajectory[0] - self.start_point)**2)
            end_diff = torch.sum((current_trajectory[-1] - self.end_point)**2)
            loss_start_end = 1000.0 * (start_diff + end_diff)

        # --- Total Weighted Loss ---
        total_loss = (self.lambda_collision * loss_collision +
                      self.lambda_length * loss_length +
                      loss_start_end)

        # --- Visualization (uses original trajectory for drawing lines) ---
        # Detach original trajectory for visualization
        trajectory_np = current_trajectory.detach().cpu().numpy()
        frame = self._draw_trajectory(trajectory_np)
        # Optional: could also draw the sub-sampled points in a different color for debugging
        # sub_points_np = sub_sampled_points.detach().cpu().numpy() # If needed
        # ... draw sub_points_np on frame ...
        self.frames.append(frame)

        return total_loss

    def get_frames(self) -> List[np.ndarray]:
        """Returns the list of generated visualization frames."""
        return self.frames

    def get_final_trajectory(self) -> np.ndarray:
        """Returns the final optimized trajectory (key points) as a numpy array."""
        return self._get_current_trajectory().detach().cpu().numpy()
# ==========================
# Example Usage
# ==========================
if __name__ == "__main__":
    # --- Create a Simple Maze ---
    # 0 = wall, 1 = free space
    maze_grid = np.ones((30, 40), dtype=np.uint8)
    maze_grid[0, :] = 0 # Top border
    maze_grid[-1, :] = 0 # Bottom border
    maze_grid[:, 0] = 0 # Left border
    maze_grid[:, -1] = 0 # Right border

    maze_grid[10:20, 15:25] = 0 # Add an obstacle block
    maze_grid[5:15, 5] = 0      # Add a vertical wall segment
    maze_grid[25, 10:30] = 0    # Add a horizontal wall segment

    # Ensure start/end are in free space
    start = (5.0, 5.0)
    end = (35.0, 25.0)
    maze_grid[int(start[1]), int(start[0])] = 1
    maze_grid[int(end[1]), int(end[0])] = 1

    # --- Setup Optimization ---
    print("Setting up optimization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    objective = MazeTrajectoryOptimization(
        maze=maze_grid,
        start_point=start,
        end_point=end,
        lambda_collision = 1000,
        n_points=30,          # More points for smoother path
        init_straight=False,
        n_subsample_points=10,
        optimize_start_end=True, # Keep start/end fixed
    )

    # --- Choose an Optimizer ---
    # Adam often works well. LBFGS can be good for smooth objectives but might need line search.
    # optimizer = torch.optim.Adam(objective.parameters(), lr=0.1)
    optimizer = torch.optim.LBFGS(objective.parameters(), lr=1, line_search_fn='strong_wolfe') # LBFGS often works well here


    # --- Optimization Loop ---
    print("Starting optimization...")
    n_steps = 100 # For Adam; LBFGS uses max_iter within closure

    if isinstance(optimizer, torch.optim.LBFGS):
        for i in range(n_steps // 10): # LBFGS does multiple iterations per step
            print(f"LBFGS Step {i+1}")
            def closure():
                optimizer.zero_grad()
                loss = objective()
                loss.backward()
                print(f"  Loss: {loss.item():.4f}")
                return loss
            optimizer.step(closure)
            if objective() < 0.1 : # Early stopping condition (optional)
                print("Loss is low, stopping early.")
                break

    else: # Example for Adam/SGD style optimizers
         for i in range(n_steps):
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Step {i+1}/{n_steps}, Loss: {loss.item():.4f}")


    print("Optimization finished.")

    # --- Get Results ---
    final_trajectory = objective.get_final_trajectory()
    print(f"Final trajectory has {len(final_trajectory)} points.")

    # --- Display Optimization Process (Frames) ---
    print("Displaying optimization frames (press any key to advance, q to quit)...")
    frames = objective.get_frames()
    render_frames('frames.mp4', frames)

    # --- Optionally save the final trajectory image ---
    final_frame = frames[-1]
    cv2.imwrite("final_trajectory.png", final_frame)
    print("Saved final trajectory image to final_trajectory.png")

    # --- Optionally save frames as a video ---
    # try:
    #     import imageio
    #     print("Saving frames as trajectory_optimization.gif...")
    #     imageio.mimsave('trajectory_optimization.gif', frames, fps=10)
    #     print("Saved trajectory_optimization.gif")
    # except ImportError:
    #     print("Install imageio (`pip install imageio`) to save animation as GIF.")