import numpy as np
import visualizer3d_sans_vbo as visualizer3d
import sys
from numba import njit, prange
from mpi4py import MPI

G = 1.560339e-13

def generate_star_color(mass):
    if mass > 5.0:
        return (150, 180, 255)
    elif mass > 2.0:
        return (255, 255, 255)
    elif mass >= 1.0:
        return (255, 255, 200)
    else:
        return (255, 150, 100)

@njit(parallel=True)
def update_stars_in_grid(cell_start_indices, body_indices, cell_masses, cell_com_positions,
                         masses, positions, grid_min, cell_size, n_cells, local_bodies):
    n_local = len(local_bodies)
    cell_counts = np.zeros(int(np.prod(n_cells)), dtype=np.int64)
    
    for i in range(n_local):
        ibody = local_bodies[i]
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for j in range(3):
            cell_idx[j] = max(0, min(n_cells[j] - 1, cell_idx[j]))
        morse_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        cell_counts[morse_idx] += 1
    
    running_index = 0
    for i in range(len(cell_counts)):
        cell_start_indices[i] = running_index
        running_index += cell_counts[i]
    cell_start_indices[len(cell_counts)] = running_index
    
    current_counts = np.zeros(int(np.prod(n_cells)), dtype=np.int64)
    for i in range(n_local):
        ibody = local_bodies[i]
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for j in range(3):
            cell_idx[j] = max(0, min(n_cells[j] - 1, cell_idx[j]))
        morse_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        index_in_cell = cell_start_indices[morse_idx] + current_counts[morse_idx]
        body_indices[index_in_cell] = ibody
        current_counts[morse_idx] += 1
    
    for i in prange(len(cell_counts)):
        cell_mass = 0.0
        com_x, com_y, com_z = 0.0, 0.0, 0.0
        start_idx = cell_start_indices[i]
        end_idx = cell_start_indices[i + 1]
        
        for j in range(start_idx, end_idx):
            ibody = body_indices[j]
            m = masses[ibody]
            cell_mass += m
            com_x += positions[ibody, 0] * m
            com_y += positions[ibody, 1] * m
            com_z += positions[ibody, 2] * m
        
        if cell_mass > 0.0:
            cell_masses[i] = cell_mass
            cell_com_positions[i, 0] = com_x / cell_mass
            cell_com_positions[i, 1] = com_y / cell_mass
            cell_com_positions[i, 2] = com_z / cell_mass
        else:
            cell_masses[i] = 0.0
            cell_com_positions[i, 0] = 0.0
            cell_com_positions[i, 1] = 0.0
            cell_com_positions[i, 2] = 0.0

@njit(parallel=True)
def compute_acceleration(positions, masses, cell_start_indices, body_indices,
                        cell_masses, cell_com_positions, grid_min, cell_size, n_cells, local_bodies):
    n_local = len(local_bodies)
    a = np.zeros_like(positions)
    R = 2
    
    for idx in prange(n_local):
        ibody = local_bodies[idx]
        pos = positions[ibody]
        cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)
        
        for i in range(3):
            cell_idx[i] = max(0, min(n_cells[i] - 1, cell_idx[i]))
        
        ix_min = max(0, cell_idx[0] - R)
        ix_max = min(n_cells[0] - 1, cell_idx[0] + R)
        iy_min = max(0, cell_idx[1] - R)
        iy_max = min(n_cells[1] - 1, cell_idx[1] + R)
        iz_min = max(0, cell_idx[2] - R)
        iz_max = min(n_cells[2] - 1, cell_idx[2] + R)
        
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                for iz in range(iz_min, iz_max + 1):
                    morse_idx = ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]
                    
                    if abs(ix - cell_idx[0]) > R or abs(iy - cell_idx[1]) > R or abs(iz - cell_idx[2]) > R:
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            direction = cell_com_positions[morse_idx] - pos
                            dist_sq = direction[0]**2 + direction[1]**2 + direction[2]**2
                            if dist_sq > 1.E-20:
                                distance = np.sqrt(dist_sq)
                                inv_dist3 = 1.0 / (dist_sq * distance)
                                a[ibody, :] += G * direction[:] * inv_dist3 * cell_mass
                    else:
                        start_idx = cell_start_indices[morse_idx]
                        end_idx = cell_start_indices[morse_idx + 1]
                        
                        for j in range(start_idx, end_idx):
                            jbody = body_indices[j]
                            if jbody != ibody:
                                direction = positions[jbody] - pos
                                dist_sq = direction[0]**2 + direction[1]**2 + direction[2]**2
                                if dist_sq > 1.E-20:
                                    distance = np.sqrt(dist_sq)
                                    inv_dist3 = 1.0 / (dist_sq * distance)
                                    a[ibody, :] += G * direction[:] * inv_dist3 * masses[jbody]
    
    return a

class SpatialGrid:
    def __init__(self, positions, nb_cells_per_dim):
        self.min_bounds = np.min(positions, axis=0) - 1.E-6
        self.max_bounds = np.max(positions, axis=0) + 1.E-6
        self.n_cells = np.array(nb_cells_per_dim, dtype=np.int64)
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        self.cell_start_indices = np.full(int(np.prod(self.n_cells)) + 1, -1, dtype=np.int64)
        self.body_indices = np.empty(positions.shape[0], dtype=np.int64)
        self.cell_masses = np.zeros(int(np.prod(self.n_cells)), dtype=np.float32)
        self.cell_com_positions = np.zeros((int(np.prod(self.n_cells)), 3), dtype=np.float32)

class NBodySystem:
    def __init__(self, filename, ncells_per_dir=(10, 10, 10)):
        positions = []
        velocities = []
        masses = []
        
        self.max_mass = 0.
        self.box = np.array([[-1.E-6, -1.E-6, -1.E-6], [1.E-6, 1.E-6, 1.E-6]], dtype=np.float64)
        
        with open(filename, "r") as fich:
            for line in fich:
                data = line.split()
                m = float(data[0])
                masses.append(m)
                positions.append([float(data[1]), float(data[2]), float(data[3])])
                velocities.append([float(data[4]), float(data[5]), float(data[6])])
                self.max_mass = max(self.max_mass, m)
                
                for i in range(3):
                    self.box[0][i] = min(self.box[0][i], positions[-1][i] - 1.E-6)
                    self.box[1][i] = max(self.box[1][i], positions[-1][i] + 1.E-6)
        
        self.positions = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32)
        self.masses = np.array(masses, dtype=np.float32)
        self.colors = [generate_star_color(m) for m in masses]
        self.grid = SpatialGrid(self.positions, ncells_per_dir)
        
    def update_positions(self, dt, local_bodies):
        local_bodies_arr = np.array(local_bodies, dtype=np.int64)
        
        a = compute_acceleration(self.positions, self.masses,
                                self.grid.cell_start_indices, self.grid.body_indices,
                                self.grid.cell_masses, self.grid.cell_com_positions,
                                self.grid.min_bounds, self.grid.cell_size, 
                                self.grid.n_cells, local_bodies_arr)
        
        self.positions += self.velocities * dt + 0.5 * a * dt * dt
        
        update_stars_in_grid(self.grid.cell_start_indices, self.grid.body_indices,
                            self.grid.cell_masses, self.grid.cell_com_positions,
                            self.masses, self.positions, self.grid.min_bounds,
                            self.grid.cell_size, self.grid.n_cells, local_bodies_arr)
        
        a_new = compute_acceleration(self.positions, self.masses,
                                    self.grid.cell_start_indices, self.grid.body_indices,
                                    self.grid.cell_masses, self.grid.cell_com_positions,
                                    self.grid.min_bounds, self.grid.cell_size,
                                    self.grid.n_cells, local_bodies_arr)
        
        self.velocities += 0.5 * (a + a_new) * dt

_comm_global = None
_positions_buffer = None

def updater_callback(dt):
    global _positions_buffer, _comm_global
    comm = _comm_global
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        counts = np.zeros(size, dtype=np.int32)
    else:
        counts = None
    
    comm.Barrier()
    return _positions_buffer

def run_master(system, box, colors, max_mass, dt):
    global _comm_global, _positions_buffer
    
    _comm_global = MPI.COMM_WORLD
    _positions_buffer = np.empty(system.positions.shape, dtype=np.float32)
    _positions_buffer[:] = system.positions
    
    col = colors
    intensity = np.clip(system.masses / max_mass, 0.5, 1.0)
    visu = visualizer3d.Visualizer3D(system.positions.copy(), col, intensity, 
                                      [[box[0][0], box[1][0]], 
                                       [box[0][1], box[1][1]], 
                                       [box[0][2], box[1][2]]])
    visu.run(updater=updater_callback, dt=dt)
    
    for _ in range(_comm_global.Get_size() - 1):
        _comm_global.recv(source=MPI.ANY_SOURCE, tag=0)

def run_slave(filename, ncells_per_dir, dt):
    system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    cells_per_rank = (system.grid.n_cells[0] + size - 1) // size
    local_ix_start = rank * cells_per_rank
    local_ix_end = min((rank + 1) * cells_per_rank, system.grid.n_cells[0])
    
    iteration = 0
    
    while True:
        local_bodies = []
        for ibody in range(len(system.positions)):
            cell_idx_x = int(np.floor((system.positions[ibody, 0] - system.grid.min_bounds[0]) / system.grid.cell_size[0]))
            cell_idx_x = max(0, min(int(system.grid.n_cells[0]) - 1, cell_idx_x))
            if local_ix_start <= cell_idx_x < local_ix_end:
                local_bodies.append(ibody)
        
        t_compute_start = MPI.Wtime()
        system.update_positions(dt, local_bodies)
        t_compute_end = MPI.Wtime()
        
        t_reduce_start = MPI.Wtime()
        comm.Allreduce(MPI.IN_PLACE, system.grid.cell_masses, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, system.grid.cell_com_positions, op=MPI.SUM)
        t_reduce_end = MPI.Wtime()
        
        t_sync_start = MPI.Wtime()
        
        local_count = np.array([len(local_bodies)], dtype=np.int32)
        counts = np.zeros(size, dtype=np.int32) if rank == 0 else None
        comm.Gather([local_count, MPI.INT], [counts, MPI.INT], root=0)
        
        if rank == 0:
            offsets = np.zeros(size, dtype=np.int32)
            offsets[1:] = np.cumsum(counts[:-1])
            total_count = int(counts.sum())
            recvbuf = np.empty((total_count, 3), dtype=np.float32)
        else:
            recvbuf = None
            offsets = None
        
        local_positions = system.positions[local_bodies].copy() if len(local_bodies) > 0 else np.empty((0, 3), dtype=np.float32)
        
        if rank == 0:
            comm.Gatherv([local_positions, MPI.FLOAT], [recvbuf, (counts * 3, offsets * 3), MPI.FLOAT], root=0)
            _positions_buffer[:] = recvbuf
        else:
            comm.Gatherv([local_positions, MPI.FLOAT], None, root=0)
        
        comm.Barrier()
        
        t_sync_end = MPI.Wtime()
        
        if iteration % 10 == 0 and rank != 0:
            print(f"[Rank {rank}] Compute: {(t_compute_end - t_compute_start)*1000:.2f}ms | "
                  f"Reduce: {(t_reduce_end - t_reduce_start)*1000:.2f}ms | "
                  f"Sync: {(t_sync_end - t_sync_start)*1000:.2f}ms | Bodies: {len(local_bodies)}", flush=True)
        
        iteration += 1
        
        comm.Barrier()
        if rank == 0:
            comm.bcast(True, root=0)
        else:
            stop_flag = comm.bcast(None, root=0)
            if not stop_flag:
                break
        
        comm.send(None, dest=0, tag=0)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    filename = "data/galaxy_1000"
    dt = 0.001
    n_cells_per_dir = (20, 20, 1)
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        dt = float(sys.argv[2])
    if len(sys.argv) > 5:
        n_cells_per_dir = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    
    if rank == 0:
        if size < 2:
            print("Au moins 2 processus MPI sont requis")
            sys.exit(1)
        system = NBodySystem(filename, ncells_per_dir=n_cells_per_dir)
        print(f"Simulation de {filename} avec dt = {dt} et grille {n_cells_per_dir}")
        run_master(system, system.box, system.colors, system.max_mass, dt)
    else:
        run_slave(filename, n_cells_per_dir, dt)
    
    MPI.Finalize()
