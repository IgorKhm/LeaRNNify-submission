from benchmarking import rand_pregenerated_benchmarks, generate_rand_spec_and_check_them
from contact_sequences_benchmarks import check_contact_sequence_RNN
if __name__ == '__main__':
    generate_rand_spec_and_check_them()
    rand_pregenerated_benchmarks(timeout=600, check_flows=True)
    check_contact_sequence_RNN()

