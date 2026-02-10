#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include <mpi.h>

void generate_random_array(std::vector<int>& arr, int size, int min_v, int max_v, char type) {
    static std::mt19937 gen(std::time(nullptr)); 
    
    arr.clear();
    arr.reserve(size);

    for (int i = 0; i < size; ++i) {
        int value;
        if (type == 'n') { // Normal
            std::normal_distribution<> d((min_v + max_v) / 2.0, (max_v - min_v) / 6.0);
            value = static_cast<int>(d(gen));
        } else if (type == 'e') { // Exponential
            std::exponential_distribution<> d(1.0 / ((min_v + max_v) / 2.0));
            value = static_cast<int>(d(gen));
        } else { // Uniforme
            std::uniform_int_distribution<> d(min_v, max_v);
            value = d(gen);
        }
        
        if (value < min_v) value = min_v;
        if (value > max_v) value = max_v;
        
        arr.push_back(value);
    }
}

void print_array(const std::vector<int>& arr) {
    for (size_t i = 0; i < arr.size(); ++i) {
        std::cout << arr[i];
        if (i < arr.size() - 1) {
            std::cout << "-";
        }
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nbp;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbp);

    int total_size = 1000000; 
    int min_val = 0;
    int max_val = 100000;
    std::vector<int> global_arr;
    int local_size = total_size / nbp;
    std::vector<int> local_arr(local_size);

    if (rank == 0) {
        generate_random_array(global_arr, total_size, min_val, max_val, 'e'); // 'n' for normal, 'e' for exponential, 'u' for uniform
        // print_array(global_arr); //optional for small sizes
    }

    // --- START GLOBAL TIMER ---
    double global_start = MPI_Wtime();

    MPI_Scatter(global_arr.data(), local_size, MPI_INT, 
                local_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Phase 2: Local Binning
    std::vector<std::vector<int>> send_buckets(nbp);
    int interval = (max_val - min_val + 1) / nbp;
    for (int num : local_arr) {
        int target = num / interval;
        if (target >= nbp) target = nbp - 1;
        send_buckets[target].push_back(num);
    }

    // Phase 3: All-to-All Exchange
    std::vector<int> send_counts(nbp), recv_counts(nbp);
    for (int i = 0; i < nbp; ++i) send_counts[i] = send_buckets[i].size();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> s_displs(nbp, 0), r_displs(nbp, 0);
    int total_recv = 0;
    for (int i = 0; i < nbp; ++i) {
        total_recv += recv_counts[i];
        if (i > 0) {
            s_displs[i] = s_displs[i-1] + send_counts[i-1];
            r_displs[i] = r_displs[i-1] + recv_counts[i-1];
        }
    }

    std::vector<int> send_buf, recv_buf(total_recv);
    for (const auto& bucket : send_buckets) {
        send_buf.insert(send_buf.end(), bucket.begin(), bucket.end());
    }

    MPI_Alltoallv(send_buf.data(), send_counts.data(), s_displs.data(), MPI_INT,
                  recv_buf.data(), recv_counts.data(), r_displs.data(), MPI_INT, MPI_COMM_WORLD);

    // Phase 4: Local Sort (Timing only the sorting part)
    double sort_start = MPI_Wtime();
    std::sort(recv_buf.begin(), recv_buf.end());
    double sort_end = MPI_Wtime();
    double local_sort_time = sort_end - sort_start;

    // Phase 5: Gather back
    std::vector<int> final_counts(nbp);
    MPI_Gather(&total_recv, 1, MPI_INT, final_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> g_displs(nbp, 0);
    if (rank == 0) {
        for (int i = 1; i < nbp; ++i) g_displs[i] = g_displs[i-1] + final_counts[i-1];
        global_arr.resize(total_size);
    }

    MPI_Gatherv(recv_buf.data(), total_recv, MPI_INT, 
                global_arr.data(), final_counts.data(), g_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    
    // --- END GLOBAL TIMER ---
    double global_end = MPI_Wtime();
    double total_local_time = global_end - global_start;

    // Phase 6: Performance Analysis (Reductions)
    double max_sort_time, max_total_time;
    MPI_Reduce(&local_sort_time, &max_sort_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_local_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // print_array(global_arr); // Optional for small sizes
        std::cout << "\n========================================" << std::endl;
        std::cout << "PERFORMANCE RESULTS (" << nbp << " cores)" << std::endl;
        std::cout << "Total Array Size: " << total_size << std::endl;
        std::cout << "Max Local Sort Time: " << max_sort_time << "s" << std::endl;
        std::cout << "Max Total Algorithm: " << max_total_time << "s" << std::endl;
        std::cout << "Communication Overhead: " << (max_total_time - max_sort_time) << "s" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }

    // Barrier to keep output clean
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << rank << " -> Elements: " << recv_buf.size() 
              << " | Sort: " << local_sort_time << "s" << std::endl;

    MPI_Finalize();
    return 0;
}