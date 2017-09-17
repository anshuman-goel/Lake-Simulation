./lake 16 5 1.0 8 > grid_comparison
./lake 32 5 1.0 8 >> grid_comparison
./lake 64 5 1.0 8 >> grid_comparison
./lake 128 5 1.0 8 >> grid_comparison
./lake 256 5 1.0 8 >> grid_comparison
./lake 512 5 1.0 8 >> grid_comparison
./lake 1024 5 1.0 8 >> grid_comparison

./lake 128 5 1.0 1 > grid_comparison_thread_128
./lake 128 5 1.0 2 >> grid_comparison_thread_128
./lake 128 5 1.0 4 >> grid_comparison_thread_128
./lake 128 5 1.0 8 >> grid_comparison_thread_128
./lake 128 5 1.0 16 >> grid_comparison_thread_128
./lake 128 5 1.0 32 >> grid_comparison_thread_128
./lake 128 5 1.0 64 >> grid_comparison_thread_128
./lake 128 5 1.0 128 >> grid_comparison_thread_128

./lake 1024 5 1.0 1 > grid_comparison_thread_1024
./lake 1024 5 1.0 2 >> grid_comparison_thread_1024
./lake 1024 5 1.0 4 >> grid_comparison_thread_1024
./lake 1024 5 1.0 8 >> grid_comparison_thread_1024
./lake 1024 5 1.0 16 >> grid_comparison_thread_1024
./lake 1024 5 1.0 32 >> grid_comparison_thread_1024
./lake 1024 5 1.0 64 >> grid_comparison_thread_1024
./lake 1024 5 1.0 128 >> grid_comparison_thread_1024
./lake 1024 5 1.0 256 >> grid_comparison_thread_1024
./lake 1024 5 1.0 512 >> grid_comparison_thread_1024
./lake 1024 5 1.0 1024 >> grid_comparison_thread_1024
