# Adapted from knu-lcbc/MolForge; licensed CC BY-NC-SA4.0.
# Only inference constants are retained; importing creates no directories.
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
num_heads = 8
num_layers = 6
dim_model = 512
dim_ff = 2048
dim_k = dim_model // num_heads
dropout_rate = 0.1
beam_size = 10
trg_seq_len = 130
