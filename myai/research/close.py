import torch

class Simulation:
    def __init__(self, objective_function, initial_coords, initial_height,
                 dt, gravity, bounciness, air_resistance, wall_threshold):
        self.objective = objective_function
        self.n = len(initial_coords)
        self.position = torch.zeros(self.n + 1)
        self.position[:self.n] = torch.tensor(initial_coords, dtype=torch.float)
        loss, grad = self.objective(self.position[:self.n])
        surface_height = loss.item()
        self.position[self.n] = surface_height + initial_height
        self.velocity = torch.zeros(self.n + 1)
        self.dt = dt
        self.g = gravity
        self.e = bounciness
        self.k = air_resistance
        self.wall_threshold = wall_threshold
        self.epsilon = 1e-6  # Small value to prevent numerical issues

    def step(self):
        # Get current surface height and gradient at current position
        current_coords = self.position[:self.n]
        loss, grad = self.objective(current_coords)
        surface_height = loss.item()

        # Determine if the ball is in contact with the surface
        in_contact = self.position[self.n] <= surface_height + self.epsilon

        if in_contact:
            # Compute normal vector: [grad_x, -1], normalized
            normal = torch.cat((grad, -torch.ones(1)))
            normal = normal / normal.norm()

            # Gravitational acceleration decomposition
            g_vector = torch.zeros_like(self.position)
            g_vector[self.n] = -self.g  # Gravity downward
            g_normal = (g_vector @ normal) * normal
            g_tangential = g_vector - g_normal

            # Update velocity based on forces
            acceleration = g_tangential.clone()
            # Air resistance opposes velocity
            air_acceleration = -self.k * self.velocity
            total_acceleration = acceleration + air_acceleration
        else:
            # Only gravity in z-direction and air resistance when in the air
            acceleration = torch.zeros_like(self.position)
            acceleration[self.n] = -self.g  # Gravity downward
            # Air resistance opposes velocity
            air_acceleration = -self.k * self.velocity
            total_acceleration = acceleration + air_acceleration

        # Update velocity using Euler integration
        self.velocity += total_acceleration * self.dt
        # Update position using Euler integration
        new_position = self.position + self.velocity * self.dt

        # Get surface height and gradient at new position
        current_coords_new = new_position[:self.n]
        loss_new, grad_new = self.objective(current_coords_new)
        surface_height_new = loss_new.item()

        # Check if the ball is below the surface with epsilon
        if new_position[self.n] < surface_height_new - self.epsilon and self.velocity[self.n] < 0:
            # Ball has gone below surface, find intersection
            x = self.position[:self.n]
            v_x = self.velocity[:self.n]
            v_z = self.velocity[self.n]
            f_x = loss.item()
            # Linear approximation: f(x + v_x * t) ≈ f(x) + grad(x) @ (v_x * t)
            # Height equation: position_z + v_z * t = f(x) + grad(x) @ (v_x * t)
            denom = v_z - torch.dot(grad, v_x) + 1e-8  # Prevent division by zero
            t = (f_x - self.position[self.n]) / denom
            if 0 <= t <= self.dt:
                # Intersection occurs at t
                intersection_pos = self.position + self.velocity * t
                # Set position to intersection_pos
                self.position = intersection_pos
                # Compute normal vector: [grad_x, -1], normalized
                normal_intersection = torch.cat((grad, -torch.ones(1)))
                normal_intersection = normal_intersection / normal_intersection.norm()
                # Reflect velocity
                v = self.velocity
                v_normal = (v @ normal_intersection) * normal_intersection
                v_tangent = v - v_normal
                # Bounce: reverse the normal component with bounciness
                v_reflected = v_tangent - self.e * v_normal
                self.velocity = v_reflected
                print("Ball bounced after flying.")
            else:
                # No intersection within the time step, set position to surface height
                self.position = new_position
                self.position[self.n] = surface_height_new
                # Project velocity onto tangent plane
                normal_new = torch.cat((grad_new, -torch.ones(1)))
                normal_new = normal_new / normal_new.norm()
                v = self.velocity
                v_normal = (v @ normal_new) * normal_new
                v_tangent = v - v_normal
                self.velocity = v_tangent
        else:
            # Ball is above surface, set position to new_position
            self.position = new_position
            # Check for wall bouncing based on gradient magnitude
            grad_norm = grad.norm()
            if grad_norm > self.wall_threshold:
                # Treat as a wall: bounce off
                normal_wall = grad / grad_norm  # Outward normal
                v = self.velocity[:self.n]
                v_normal = (v @ normal_wall) * normal_wall
                v_tangent = v - v_normal
                # Bounce: reverse the normal component with bounciness
                v_reflected = v_tangent - self.e * v_normal
                self.velocity[:self.n] = v_reflected
                print("Ball bounced after hitting a wall.")
            # Check if the ball is flying into the air
            if self.velocity[self.n] > 0:
                print("Ball is flying into the air.")

    def get_position(self):
        return self.position.clone()

    def set_position(self, new_position):
        self.position = new_position.clone()

# Example usage:
# Define an objective function, for example, a sphere function
def sphere_function(x):
    loss = torch.sum(x**2)
    grad = 2 * x
    return loss, grad

# Initialize simulation parameters
initial_coords = torch.tensor([1.0, 1.0])  # 2D example
initial_height = 0.0  # Starting above the surface
dt = 0.001  # Reduced time step for stability
gravity = 9.81
bounciness = 0.8
air_resistance = 0.1
wall_threshold = 10.0  # Arbitrary threshold

# Create simulation instance
sim = Simulation(sphere_function, initial_coords, initial_height,
                 dt, gravity, bounciness, air_resistance, wall_threshold)


# Run simulation steps
for _ in range(400):
    print(_)
    print(f'{sim.position}, {sim.velocity}')
    sim.step()
    # Optionally, print or record position
    # print(sim.get_position())