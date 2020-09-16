from benchmarking import rand_pregenerated_benchmarks, generate_rand_spec_and_check_them

if __name__ == '__main__':
    generate_rand_spec_and_check_them()
    rand_pregenerated_benchmarks(timeout=600, check_flows=True)
