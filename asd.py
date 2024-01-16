from mpi4py import MPI
import numpy as np

def distributed_vertex_coloring(rank, neighbors, num_nodes, comm):
    colors = [-1] * num_nodes
    rounds = num_nodes  # Her bir düğüm için bir tur

    for round_num in range(1, rounds + 1):
        max_rank_node = max(neighbors + [rank])

        if max_rank_node == rank:
            selected_color = max(set(range(rounds)) - set(colors))
            colors[rank] = selected_color
            print(f"Round {round_num}: rank: {rank}, color: {selected_color}")

        comm.Barrier()

        selected_color = comm.bcast(selected_color, root=max_rank_node)

        for neighbor in neighbors:
            received_color = np.empty(1, dtype=np.int32)
            comm.Recv([received_color, MPI.INT], source=neighbor)
            received_color_value = received_color[0]
            colors[neighbor] = received_color_value

        comm.Barrier()

        print(f"Round {round_num}: rank: {rank}, selected_color: {selected_color}, colors: {colors}")

        # Seçilen rengi diğer düğümlere iletmek
        for neighbor in neighbors:
            comm.Send([np.array([selected_color], dtype=np.int32), MPI.INT], dest=neighbor)

    print(f"Final state of rank {rank}: {colors}")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    try:
        connections = {0: [3, 4, 7], 1: [7, 5], 2: [6, 3, 4], 3: [6, 2, 4, 0], 4: [2, 3, 0, 7, 5], 5: [4, 7, 1], 6: [3, 2], 7: [0, 4, 5, 1]}
        neighbors = connections.get(rank, [])

        num_nodes = size

        distributed_vertex_coloring(rank, neighbors, num_nodes, comm)

    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        comm.Abort()

if __name__ == "__main__":
    main()
