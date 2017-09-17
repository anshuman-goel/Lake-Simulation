prun ./lake 16 5 1.0 8 > mpi_grid_comparison
prun ./lake 32 5 1.0 8 >> mpi_grid_comparison
prun ./lake 64 5 1.0 8 >> mpi_grid_comparison
prun ./lake 128 5 1.0 8 >> mpi_grid_comparison
prun ./lake 256 5 1.0 8 >> mpi_grid_comparison
prun ./lake 512 5 1.0 8 >> mpi_grid_comparison
prun ./lake 1024 5 1.0 8 >> mpi_grid_comparison

prun ./lake 128 5 1.0 1 > mpi_grid_comparison_thread_128
prun ./lake 128 5 1.0 2 >> mpi_grid_comparison_thread_128
prun ./lake 128 5 1.0 4 >> mpi_grid_comparison_thread_128
prun ./lake 128 5 1.0 8 >> mpi_grid_comparison_thread_128
prun ./lake 128 5 1.0 16 >> mpi_grid_comparison_thread_128
prun ./lake 128 5 1.0 32 >> mpi_grid_comparison_thread_128
prun ./lake 128 5 1.0 64 >> mpi_grid_comparison_thread_128
prun ./lake 128 5 1.0 128 >> mpi_grid_comparison_thread_128

prun ./lake 1024 5 1.0 1 > mpi_grid_comparison_thread_1024
prun ./lake 1024 5 1.0 2 >> mpi_grid_comparison_thread_1024
prun ./lake 1024 5 1.0 4 >> mpi_grid_comparison_thread_1024
prun ./lake 1024 5 1.0 8 >> mpi_grid_comparison_thread_1024
prun ./lake 1024 5 1.0 16 >> mpi_grid_comparison_thread_1024
prun ./lake 1024 5 1.0 32 >> mpi_grid_comparison_thread_1024
prun ./lake 1024 5 1.0 64 >> mpi_grid_comparison_thread_1024
prun ./lake 1024 5 1.0 128 >> mpi_grid_comparison_thread_1024
prun ./lake 1024 5 1.0 256 >> mpi_grid_comparison_thread_1024
prun ./lake 1024 5 1.0 512 >> mpi_grid_comparison_thread_1024
prun ./lake 1024 5 1.0 1024 >> mpi_grid_comparison_thread_1024
