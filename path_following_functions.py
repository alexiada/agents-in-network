import numpy as np
np.random.seed(42)
import numba
numba.config.DEFAULT_FLOAT_TYPE = numba.float32

##################################### INIZIALIZE ###############################
def initialize(node, site, site_probabilities, connectivity_matrix, MAX_STEPS_IN_PATH, NUM_PARTICLES, MAX_FRIENDS):
    velocity = np.random.uniform(-1, 1, size=(NUM_PARTICLES, 2)).astype(np.float32)
    groups = create_groups(NUM_PARTICLES, MAX_FRIENDS)
    nodes_in_path, steps_in_path,\
    path = generate_initial_paths(node, site, site_probabilities, connectivity_matrix, 
                                  MAX_STEPS_IN_PATH, NUM_PARTICLES, groups)
    position = assign_group_positions(NUM_PARTICLES, node, groups, nodes_in_path, path)
    return position, velocity, nodes_in_path, steps_in_path, path, groups

def create_groups(NUM_PARTICLES, MAX_FRIENDS):
    groups_list = []
    index = 0
    
    while index < NUM_PARTICLES:
        num_friends = np.random.randint(2, MAX_FRIENDS + 1)  # Random group size (between 2 and MAX_FRIENDS)
        
        if index + num_friends > NUM_PARTICLES:
            num_friends = NUM_PARTICLES - index  # Adjust last group size to fit remaining particles

        group = list(np.arange(index, index + num_friends))  # Assign consecutive particle IDs
        groups_list.append(group)
        index += num_friends

    # Update actual group parameters
    NUM_GROUPS = len(groups_list)
    MAX_FRIENDS = max(len(g) for g in groups_list)

    # Convert to an array, padding with -1 where necessary
    groups = -np.ones((NUM_GROUPS, MAX_FRIENDS + 1), dtype=int)
    for i, group in enumerate(groups_list):
        groups[i, :len(group)] = group  # Assign particle IDs
        groups[i, -1] = len(group)  # Store group size in last element

    return groups

def map_groups_and_particles(groups, NUM_PARTICLES):
    group_association = np.full(NUM_PARTICLES, -1, dtype=np.int32)
    for group_idx in range(groups.shape[0]):  
        num_valid = groups[group_idx, -1]  # Number of valid particles in this group
        particle_ids = groups[group_idx, :num_valid]  # Indices of particles in the group
        group_association[particle_ids] = group_idx  # Assign all members the same group index
    return group_association

def generate_initial_paths(node, site, site_probabilities, connectivity_matrix, MAX_STEPS_IN_PATH, NUM_PARTICLES, groups):
    NUM_NODES = node.shape[0]
    nodes_in_path = np.full((NUM_PARTICLES, MAX_STEPS_IN_PATH), -1, dtype=np.int32)
    steps_in_path = np.zeros(NUM_PARTICLES, dtype=np.int32)
    path = np.zeros((NUM_PARTICLES, MAX_STEPS_IN_PATH, 2), dtype=np.float32)

    for group_idx in range(groups.shape[0]):
        num_valid = groups[group_idx, -1]  # Number of valid particles in group
        particle_ids = groups[group_idx, :num_valid]  # Get particle indices in this group

        start = np.random.choice(np.setdiff1d(np.arange(NUM_NODES), site))
        target = np.random.choice(site, p=site_probabilities)  

        group_nodes, group_steps, group_path = gbfs(start, target, node, site, connectivity_matrix, MAX_STEPS_IN_PATH)
        
        for particle in particle_ids:
            nodes_in_path[particle] = group_nodes.copy()
            steps_in_path[particle] = group_steps
            path[particle] = group_path.copy()
    
    return nodes_in_path, steps_in_path, path


def assign_group_positions(NUM_PARTICLES, node, groups, nodes_in_path, path):
    position = np.empty((NUM_PARTICLES, 2), dtype=np.float32)

    for group_idx in range(groups.shape[0]):
        num_valid = groups[group_idx, -1]  # Number of valid particles in group
        particle_ids = groups[group_idx, :num_valid]  # Get particle indices in this group

        # Assign group centre to the first node in the path of the first particle
        start_node = nodes_in_path[particle_ids[0], 0]
        group_centre = node[start_node].copy()

        # Move the centre up to 75% toward the next node
        next_node_position = path[particle_ids[0], 1]  # Use the first particle's path
        displacement_vector = (next_node_position - group_centre) * 0.75
        shift = np.random.uniform(0, 1) * displacement_vector  # Compute the shift once

        # Apply the same shift to all particles
        group_centre += shift

        # Spread the group members around the centre while keeping them together
        jitter = np.random.uniform(-6, 6, (num_valid, 2)).astype(np.float32)
        position[particle_ids] = group_centre + jitter

    return position


########################################################################################
############################## NETWORK #############################################
@numba.jit(nopython=True)
def get_connection_index(i, j, connections, connections_start_idx):
    """
    Finds the index k for (i, j) by scanning only relevant connections.
    """
    if i > j:
        i, j = j, i  # Ensure i < j

    start = connections_start_idx[i]
    if start == -1:  # If i has no connections
        return -1

    # Scan forward to find (i, j)
    while start < len(connections) and connections[start, 0] == i:
        if connections[start, 1] == j:
            return start  # Found the correct connection
        start += 1

    return -1  # Not found

@numba.jit(nopython=True, parallel=False, fastmath=True)
def gbfs(start, target, node, site, connectivity_matrix,
         MAX_STEPS_IN_PATH):
    #path finding algorithm based on Greedy Best-First Search
    nodes_in_path = np.full(MAX_STEPS_IN_PATH, -1, dtype=np.int32)
    path_length = 0

    num_nodes = connectivity_matrix.shape[0]
    
    # Precompute heuristics (straight-line distance to the target)
    heuristics = (node[:, 0] - node[target, 0]) ** 2 + (node[:, 1] - node[target, 1]) ** 2
    
    # Initialise visited array with all cul-de-sac nodes marked as visited except the target
    visited = np.zeros(num_nodes, dtype=numba.boolean)
    for i in range(len(site)):
        if site[i] != target:
            visited[site[i]] = True
    visited[start] = True  # Mark the start node as visited
    
    # Initialise path tracking
    nodes_in_path[0] = start
    path_length = 1  # Number of steps in the current path
    
    current_node = start
    while current_node != target:
        # Find unvisited connected nodes
        unvisited_nodes = np.empty(num_nodes, dtype=numba.int32)
        unvisited_count = 0
        for i in range(num_nodes):
            if connectivity_matrix[current_node, i] and not visited[i]:
                unvisited_nodes[unvisited_count] = i
                unvisited_count += 1
        
        # If no unvisited nodes remain, terminate search
        if unvisited_count == 0:
            nodes_in_path[path_length] = -1  # Mark as invalid
            path_length = 0  # Path length 0 indicates failure
            print(f"gbfs failure between {start} and {target}")
            return nodes_in_path, path_length, np.zeros((MAX_STEPS_IN_PATH, 2), dtype=np.float32)
        
        # Select the node with the smallest heuristic value
        next_node = -1
        min_heuristic = np.inf
        for i in range(unvisited_count):
            node_idx = unvisited_nodes[i]
            heuristic = heuristics[node_idx]
            if heuristic < min_heuristic:
                min_heuristic = heuristic
                next_node = node_idx
        
        # Add the node to the path
        if path_length >= MAX_STEPS_IN_PATH - 1:  # Check to prevent overflow
            path_length = 0  # Path length 0 indicates failure
            print("Too many steps in path. Increase MAX_STEPS_IN_PATH.")
            return nodes_in_path, path_length, np.zeros((MAX_STEPS_IN_PATH, 2), dtype=np.float32)
        
        nodes_in_path[path_length] = next_node
        visited[next_node] = True
        path_length += 1   
        # Move to the next node
        current_node = next_node
        
    # Convert node indices to coordinates
    path = np.full((MAX_STEPS_IN_PATH, 2), -999., dtype=np.float32)  # Use -999 for invalid steps
    for i in range(path_length):
        node_idx = nodes_in_path[i]
        path[i] = node[node_idx]
    
    return nodes_in_path, path_length, path

# particles go from t[i] to t[i+1]
########################################################################################
############################### FORCES #################################################
@numba.jit(nopython=False, parallel=False, fastmath=True) ## This MUST be parallel=False 
def sum_all_forces(position, velocity, path, status, t, neighbor_list,
                  r_advance, K_advance,
                  H_wall, K_wall,
                  H_inside, K_inside,
                  damping_coefficient,
                  d_separation, K_separation,
                  groups, K_alignment, K_cohesion):

    force = np.zeros_like(position)
    
    # Moving particles (status > 0): Follow the path and interact with walls
    valid_moving = status > 0
    force[valid_moving] += force_towards_next_node(
        position[valid_moving], r_advance, K_advance, path[valid_moving], t[valid_moving]
    )
    # Apply alignment and cohesion forces directly since they already filters valid groups
    force += alignment_force(velocity, groups, status, K_alignment)
    force += cohesion_force(velocity, groups, status, K_cohesion)
    force_wall, path_fraction = wall_force(position[valid_moving], H_wall, K_wall, path[valid_moving], t[valid_moving])
    force[valid_moving] += force_wall 
    force_damping = np.empty_like(velocity[valid_moving])  # Create output array
    arrival_damping_force(velocity[valid_moving], path_fraction, np.float32(0.1), force_damping) # probably unecessary
    force[valid_moving] += force_damping  

    # Inside particles (status == 0): Experience inside forces and damping
    valid_inside = status == 0
    force[valid_inside] += inside_force(position[valid_inside], H_inside, K_inside, path[valid_inside], t[valid_inside])
    force[valid_inside] += damping_force(velocity[valid_inside], damping_coefficient)
    
    # Exited particles (status == -1): Only damping (to prevent floating point errors)
    valid_out = status == -1
    force[valid_out] += damping_force(velocity[valid_out], damping_coefficient)
    
    #Interaction force is always applied to all particles (prevents overlap & division by zero)
    force += separation_force(position, neighbor_list, d_separation, K_separation)
   
    return force

@numba.jit(nopython=True, parallel=False, fastmath=True) # This MUST be parallel=False 
def build_neighbor_list(position, status, CUT_OFF_LIST_SQ, MAX_NEIGH):
    NUM_PARTICLES = position.shape[0]
    neighbor_list = np.full((NUM_PARTICLES, MAX_NEIGH + 1), -1, dtype=np.int32)
    
    for i in numba.prange(NUM_PARTICLES):  # Loop over each particle
        if status[i] == -1:  # Skip exited particles completely
            continue
        
        n = 0
        for j in range(i+1, NUM_PARTICLES):
            if status[j] == -1:  # Do not include exited particles in the list
                continue

            dx = position[i, 0] - position[j, 0]
            dy = position[i, 1] - position[j, 1]
            dr2 = dx * dx + dy * dy
            
            if dr2 < CUT_OFF_LIST_SQ:
                neighbor_list[i, n] = j
                n += 1
                if n >= MAX_NEIGH:  # Check to prevent overflow
                    print(f"Too many neighbours for particle {i}. Increase MAX_NEIGH.")
                    break
        
        neighbor_list[i, -1] = n  # Store the number of neighbours for particle i

    return neighbor_list

@numba.jit(nopython=True, parallel=False, fastmath=True)
def separation_force(position, neighbor_list, d_separation, K_separation):
    num_particles = position.shape[0]
    force = np.zeros_like(position)
    d_separation_squared = d_separation ** 2  # Precompute squared minimum distance

    # Compute pairwise forces based on the neighbour list
    for i in numba.prange(num_particles):  # Parallel loop over particles
        n = neighbor_list[i, -1]  # Number of neighbours for particle i
        for k in range(n):
            j = neighbor_list[i, k]  # Get the neighbour index

            # Vector between particles i and j
            delta = position[j] - position[i]
            distance_squared = delta[0]**2 + delta[1]**2  # Squared distance

            if distance_squared < d_separation_squared: 
                distance = np.sqrt(distance_squared)  # Compute actual distance
                # Compute the repulsion force magnitude (F ∝ 1 / distance^2)
                force_magnitude = K_separation / distance_squared
                # Normalize the direction
                normalized_delta = delta / distance
                repulsion_force = force_magnitude * normalized_delta

                # Apply equal and opposite forces
                force[i] -= repulsion_force
                force[j] += repulsion_force

    return force

@numba.jit(nopython=True, parallel=False, fastmath=True)
def alignment_force(velocity, groups, status, K_alignment):
    force = np.zeros_like(velocity)
    for group_idx in range(groups.shape[0]): 
        num_valid = groups[group_idx, -1]  # Number of valid particles in group
        particle_ids = groups[group_idx, :num_valid]  # Indices of particles in the group

        # Filter only moving groups
        if np.all(status[particle_ids] > 0):
            avg_velocity = np.zeros(2, dtype=np.float32)         
            for p in particle_ids:
                avg_velocity += velocity[p]
            avg_velocity /= num_valid
            # Adjust velocities towards the group average for moving particles
            for p in particle_ids:
                force[p] = K_alignment * (avg_velocity - velocity[p])   
    return force

@numba.jit(nopython=True, parallel=False, fastmath=True)
def cohesion_force(position, groups, status, K_cohesion):
    force = np.zeros_like(position)
    for group_idx in range(groups.shape[0]): 
        num_valid = groups[group_idx, -1]  # Number of valid particles in group
        particle_ids = groups[group_idx, :num_valid]  # Indices of particles in the group

        # Apply cohesion only if ALL particles in the group are moving
        if np.all(status[particle_ids] > 0):
            avg_position = np.zeros(2, dtype=np.float32)
            
            for p in particle_ids:
                avg_position += position[p]
            avg_position /= num_valid  # Compute center of mass
            
            # Apply force towards the group center
            for p in particle_ids:
                force[p] = K_cohesion * (avg_position-position[p])  # Pull particles toward the center
    
    return force


@numba.jit(nopython=True, parallel=False, fastmath=True)
def force_towards_next_node(position, r_advance, K_advance, path, t):
    num_particles = position.shape[0]
    force = np.zeros_like(position)
    for i in numba.prange(num_particles):
        r = path[i, t[i] + 1] - path[i, t[i]] #edge length
        r_squared = np.dot(r, r)
        r_norm = np.sqrt(r_squared)  
        r_normalized = r / r_norm
        b = position[i] + r_advance * r_normalized
        force[i] = K_advance * (b - position[i])
    return force

@numba.jit(nopython=True, parallel=False, fastmath=True)
def wall_force(position, H_wall, K_wall, path, t):
    num_particles = position.shape[0]
    H_wall_sq = H_wall**2
    force = np.zeros_like(position)
    path_fraction = -np.ones(num_particles, dtype=np.float32)
    for i in numba.prange(num_particles):
        p = position[i] - path[i, t[i]] #position vector
        r = path[i, t[i] + 1] - path[i, t[i]] #edge vector
        proj = np.dot(p,r)/np.dot(r,r)
        path_fraction[i] = proj
        h = proj * r - p
        h_mag_sq = np.dot(h,h)
        if h_mag_sq > H_wall_sq:
            force[i] = K_wall * h * h_mag_sq
    return force, path_fraction

@numba.jit(nopython=True, parallel=False, fastmath=True)
def inside_force(position, H_inside, K_inside, path, t):
    num_particles = position.shape[0]
    force = np.zeros_like(position)
    inv_H_inside = 1.0 / H_inside  # Precompute inverse of H_wall

    for i in numba.prange(num_particles):
        current_node_index = t[i]
        current_node_position = path[i, current_node_index]
        
        # Compute displacement from current position to the next node
        displacement = current_node_position - position[i]
        normalised_displacement = np.abs(displacement) * inv_H_inside 
        squircle_dist = np.sqrt(np.sqrt(normalised_displacement[0]**4 + normalised_displacement[1]**4))
        # Apply force if the particle is outside the squircle boundary
        if squircle_dist > 1:
            displacement_squared = np.dot(displacement, displacement)
            displacement_norm = np.sqrt(displacement_squared)
            displacement_direction = displacement / displacement_norm
            scaled_displacement = (squircle_dist - 1) * displacement_direction
            force[i] = K_inside * scaled_displacement

    return force

@numba.vectorize(["float32(float32, float32)"], target="cpu", fastmath=True)
def damping_force(velocity, damping_coefficient):
    return -damping_coefficient * velocity

@numba.guvectorize(
    ["void(float32[:], float32, float32, float32[:])"],  
    "(d),(),()->(d)", target="cpu", nopython=True, fastmath=True)
def arrival_damping_force(velocity, path_fraction, damping_coefficient, force_out):
    # Damping factor for particles near nodes
    path_funz = min(1, 1024.*(path_fraction-0.5)**8)
    damping = damping_coefficient * path_funz  
    force_out[:] = -damping * velocity  

@numba.jit(nopython=True, parallel=False, fastmath=True)
def cap_velocity(velocity, VMAX):
    num_particles = velocity.shape[0]
    VMAX_SQUARE = VMAX**2
    for i in range(num_particles): 
        norm_squared = np.dot(velocity[i], velocity[i])
        if norm_squared > VMAX_SQUARE:
            norm = np.sqrt(norm_squared)
            scale_factor = VMAX / norm
            velocity[i] *= scale_factor
    return velocity
###############################################################################
########################### INCIDENTS #####################################
@numba.jit(nopython=True, parallel=False, fastmath=True)
def compute_incidents(position, status, neighbor_list, time_spent_inside, group_association,
                      current_time, incidents, incident_counter, T0, alpha):

    num_particles = position.shape[0]
    MAX_INCIDENTS = incidents.shape[0]

    for i in numba.prange(num_particles):  
        if status[i] <= 0:  # Ignore inside (-1) or inactive particles
            continue
        
        num_neighbors = neighbor_list[i, -1]  # Number of neighbours
        for k in range(num_neighbors):
            j = neighbor_list[i, k]  # Get neighbour index
            
            if status[j] <= 0 or group_association[i] == group_association[j]:  
                continue  # Ignore inactive particles and friends

            # Compute intoxication sum
            S = np.float32(time_spent_inside[i] + time_spent_inside[j])  # Combined intoxication
            # Compute dynamic time for incidents
            T_incident = T0 / (1 + alpha * S)  # Time decreases as S increases

            # Compute probability of an incident using exponential waiting time
            P_incident = (1 - np.exp(-1. / T_incident))  # Exponential CDF

            # Draw a uniform random number and check if an incident occurs
            if np.random.uniform(0, 1) < P_incident:  
                if incident_counter < MAX_INCIDENTS:  # Ensure we don't exceed array size
                    incidents[incident_counter, 0] = position[i, 0]  # x position
                    incidents[incident_counter, 1] = position[i, 1]  # y position
                    incidents[incident_counter, 2] = np.float32(current_time)    # time of incident
                    incident_counter += 1
                else:
                    print("Warning: Incident storage full. Increase MAX_INCIDENTS.")
                    break  # Stop recording more incidents if full

    return incident_counter
###############################################################################
################################### SIMULATION #################################

def path_following_simulation(node, site, site_probabilities, exits,
                              connectivity_matrix, 
                              MAX_STEPS_IN_PATH, NUM_PARTICLES, 
                              TIME_STEPS, N_OUT, dt, VMAX, SHELL, 
                              MAX_NEIGH, CUT_OFF_LIST_SQ, 
                              r_advance, K_advance, 
                              H_wall, K_wall, 
                              H_inside, K_inside, 
                              damping_coefficient, 
                              d_separation, K_separation,
                              K_alignment, K_cohesion,
                              MAX_FRIENDS, TIME_INSIDE):
    
    # Initialize particles using the updated initialize() function
    position, velocity, nodes_in_path, steps_in_path, path, groups = initialize(node, site, site_probabilities, 
                                                                                connectivity_matrix, 
                                                                                MAX_STEPS_IN_PATH, 
                                                                                NUM_PARTICLES,
                                                                                MAX_FRIENDS)

    # Preallocate memory for full simulation data
    num_saved_steps = TIME_STEPS // N_OUT
    full_simulation = np.zeros((num_saved_steps, NUM_PARTICLES, 2), dtype=np.float32)
    full_simulation_status = np.zeros((num_saved_steps, NUM_PARTICLES), dtype=np.int32)
    full_current_site = np.zeros((num_saved_steps, NUM_PARTICLES), dtype=np.int32)
    saved_step_index = 0

    # Initialize simulation variables
    t = np.zeros(NUM_PARTICLES, dtype=np.int32)  # Independent t for each particle
    status = np.ones(NUM_PARTICLES, dtype=np.int32)  # Status: 1 = moving, 0 = inside
    # Initialize `current_site` array
    current_site = np.full(NUM_PARTICLES, -1, dtype=np.int32)
    time_spent_inside = np.zeros(NUM_PARTICLES, dtype=np.int32)
    group_association = map_groups_and_particles(groups, NUM_PARTICLES)
    incident_counter = 0
    incidents = np.zeros((1000, 3), dtype=np.float32)
    # Calculate the refresh rate for the neighbour list
    refresh_rate = int(0.5 * SHELL / (VMAX * dt))
    # transition probabilities: see update_status()
    P_move = 1. - np.exp(-1./ (2.*TIME_INSIDE))
    P_exit_given_move = 1 - np.exp(-1./ (2.*TIME_STEPS))

    # Initial neighbour list
    neighbor_list = build_neighbor_list(position, status, CUT_OFF_LIST_SQ, MAX_NEIGH)

    # Compute the initial force using the neighbour list
    force = sum_all_forces(position, velocity, path, status, t, neighbor_list,
                          r_advance, K_advance,
                          H_wall, K_wall,
                          H_inside, K_inside,
                          damping_coefficient,
                          d_separation, K_separation,
                          groups, K_alignment, K_cohesion)

    # Main simulation loop
    for i in range(TIME_STEPS):
        position, velocity, force = apply_forces_and_update_motion(position, velocity, force, dt, VMAX, 
                                                                   status, path, t, neighbor_list,
                                                                   r_advance, K_advance,
                                                                   H_wall, K_wall,
                                                                   H_inside, K_inside,
                                                                   damping_coefficient,
                                                                   d_separation, K_separation,
                                                                   groups, K_alignment, K_cohesion)

        t, status = move_along_path_and_update_status(t, position, path, nodes_in_path, 
                                                      H_wall, steps_in_path, status, node,
                                                      site, exits, connectivity_matrix,
                                                      current_site, MAX_STEPS_IN_PATH)
        
        status = update_path_based_on_status(status, P_move, P_exit_given_move, 
                                             node, site, site_probabilities, exits,
                                             path, t, nodes_in_path, steps_in_path,
                                             connectivity_matrix, current_site, 
                                             MAX_STEPS_IN_PATH, groups)
        time_spent_inside[status==0] += 1
        current_time = i
        incident_counter = compute_incidents(position, status, neighbor_list, time_spent_inside, 
                                             group_association, current_time, incidents,
                                             incident_counter, T0=1000.*TIME_STEPS, alpha=0.1)

        if i % refresh_rate == 0:
            neighbor_list = build_neighbor_list(position, status, CUT_OFF_LIST_SQ, MAX_NEIGH)

        if i % N_OUT == 0:
            full_simulation[saved_step_index] = position
            full_simulation_status[saved_step_index] = status
            full_current_site[saved_step_index] = current_site
            saved_step_index += 1

    return full_simulation, full_simulation_status, full_current_site, time_spent_inside, incidents
'''
Particle States
1.	State 0 (Inside):
  o	The particle is inside a site and does not move until it decides to leave.
  o	When it leaves the site, it can either:
     	Move to another site (State 1: Moving).
     	Move toward an exit (State 2: Exiting).
2.	State 1 (Moving):
  o	The particle is traveling from one site to another.
  o	Once it reaches the target site, it transitions back to State 0 (Inside).
3.	State 2 (Exiting):
  o	The particle is moving toward an exit (leaving the simulation box).
  o	Once it reaches the exit, it transitions to State 3 (Out).
4.	State -1 (Out):
  o	The particle has exited the simulation box.
  o	It no longer moves or updates its position.
________________________________________
How Particles Change State
1.	From State 0 (Inside):
  o	The particle decides to leave the site with probability: 1 - exp(-dt / TIME_INSIDE).
  o	If it leaves, it then decides:
     	With probability 1 - exp(-dt / TIME_STEPS), it moves toward an exit → State 2 (Exiting).
     	Otherwise, it moves toward another site → State 1 (Moving).
2.	From State 1 (Moving):
  o	If the particle reaches its target site, it transitions to State 0 (Inside).
3.	From State 2 (Exiting):
  o	If the particle reaches an exit, it transitions to State -1 (Out).
4.	From State -1 (Out):
  o	The particle does not change state anymore.

'''
@numba.jit(nopython=True, parallel=False, fastmath=True)
def apply_forces_and_update_motion(position, velocity, force, dt, VMAX, 
                                 status, path, t, neighbor_list,
                                 r_advance, K_advance,
                                 H_wall, K_wall,
                                 H_inside, K_inside,
                                 damping_coefficient,
                                 d_separation, K_separation,
                                 groups, K_alignment, K_cohesion):
    num_particles = position.shape[0]

    # Half-step velocity update
    for i in numba.prange(num_particles):
        velocity[i] += 0.5 * force[i] * dt

    valid_moving = status > 0  # status > 0 are moving particles
    velocity[valid_moving] = cap_velocity(velocity[valid_moving], VMAX)

    # Update positions
    for i in numba.prange(num_particles):
        position[i] += velocity[i] * dt

    force = sum_all_forces(position, velocity, path, status, t, neighbor_list,
                  r_advance, K_advance,
                  H_wall, K_wall,
                  H_inside, K_inside,
                  damping_coefficient,
                  d_separation, K_separation,
                  groups, K_alignment, K_cohesion)

    # Full-step velocity update
    for i in numba.prange(num_particles):
        velocity[i] += 0.5 * force[i] * dt

    valid_moving = status > 0
    velocity[valid_moving] = cap_velocity(velocity[valid_moving], VMAX)

    return position, velocity, force

@numba.jit(nopython=True)
def weighted_choice(available_targets, probabilities):
    """Numba-compatible weighted random choice."""
    rand = np.random.random()
    cumulative = 0.0
    for i in range(len(probabilities)):
        cumulative += probabilities[i]
        if rand < cumulative:
            return available_targets[i]
    return available_targets[-1]  # Fallback to the last target


@numba.jit(nopython=True, parallel=False, fastmath=True)
def generate_next_path(nodes_in_path, steps_in_path, 
                       node, target_nodes, site, site_probabilities,
                       connectivity_matrix, MAX_STEPS_IN_PATH, i):
    
    # Get the last node of the current path
    last_node_index = nodes_in_path[i, steps_in_path[i] - 1]

    # Find available targets (exclude the last node)
    available_mask = target_nodes != last_node_index
    available_targets = target_nodes[available_mask]
    available_probs = site_probabilities[available_mask]

    # Check if there are no available targets
    if len(available_targets) == 0:
        print("ERROR: No available targets to choose from.")
        raise ValueError("No available targets to choose from.")

    # Normalize probabilities for available targets
    total = available_probs.sum()

    normalized_probs = available_probs / total

    # Choose target node based on site probabilities
    target_node = weighted_choice(available_targets, normalized_probs)

    # Generate new path using GBFS
    new_nodes_in_path, new_steps_in_path, new_path = gbfs(
        last_node_index, target_node, 
        node, site, connectivity_matrix, 
        MAX_STEPS_IN_PATH
    )
    
    return new_nodes_in_path, new_steps_in_path, new_path

@numba.jit(nopython=True, parallel=False, fastmath=True)
def move_along_path_and_update_status(t, position, path, nodes_in_path, H_wall, 
                    steps_in_path, status, node, site, exits,
                    connectivity_matrix, current_site, MAX_STEPS_IN_PATH):
    num_particles = position.shape[0]
    H_wall_squared = H_wall**2
    for i in numba.prange(num_particles):
        if status[i] > 0:  # Only update moving (1) and exiting (2) particles
            next_node = path[i, t[i] + 1]  # The particle is moving towards next_node
            dx = position[i, 0] - next_node[0]
            dy = position[i, 1] - next_node[1]
            distance_squared = dx**2 + dy**2  # Avoid sqrt for efficiency
            
            # Check if particle has arrived at next_node
            if distance_squared < H_wall_squared:
                t[i] += 1  # Increment path step_counter to move to the next node
                if status[i] == 1:  # Moving between sites
                    for s in site: # checks if it arrived at a site
                        if next_node[0] == node[s, 0] and next_node[1] == node[s, 1]:
                            status[i] = 0  # Change status to 'inside'
                            current_site[i] = s  # Store the actual site index
                            break  # No need to check further

                elif status[i] == 2:  # Exiting the simulation
                    for e in exits:
                        if next_node[0] == node[e, 0] and next_node[1] == node[e, 1]:
                            status[i] = -1  # Mark as out
                            break  
    return t, status

@numba.jit(nopython=True, parallel=False, fastmath=True)
def update_path_based_on_status(status, P_move, P_exit_given_move, node, site, site_probabilities,
                                exits, path, t, nodes_in_path, steps_in_path, connectivity_matrix, 
                                current_site, MAX_STEPS_IN_PATH, groups):
    
    for group_idx in range(groups.shape[0]):
        num_valid = groups[group_idx, -1]  # Number of valid particles in group
        particle_ids = groups[group_idx, :num_valid]  # Get particle indices in this group
        
        # Check if all particles in the group are inside a site
        if np.all(status[particle_ids] == 0):  
            r = np.random.uniform()

            if r < P_move:  # Group leaves the site
                status[particle_ids] = 1  # Groups moves to another site
                current_site[particle_ids] = -1  # Reset current site
                
                if r < P_exit_given_move:  
                    status[particle_ids] = 2  # Exiting the box#

                t[particle_ids] = 0  # Reset the path counter
                
                # Select the first particle in the group to generate the path
                i = particle_ids[0]  
                target_nodes = site if status[i] == 1 else exits  # Choose new target
                
                # Generate a new path for the group
                group_nodes, group_steps, group_path = generate_next_path(
                    nodes_in_path, steps_in_path, node, 
                    target_nodes, site, site_probabilities,
                    connectivity_matrix, MAX_STEPS_IN_PATH, i
                )
                
                # Assign the same path to all particles in the group
                for particle in particle_ids:
                    nodes_in_path[particle] = group_nodes.copy()
                    steps_in_path[particle] = group_steps
                    path[particle] = group_path.copy()
                    
    return status