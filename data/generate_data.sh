# python symbol_make.py --filter_size 4 --mod_scheme "QPSK" --gen_seq_len 1300 --data_gen_type 'train' --rand_seed_train 9913
# python symbol_make.py --filter_size 16 --mod_scheme "QPSK" --gen_seq_len 2500 --data_gen_type 'train' --rand_seed_train 9925

# python symbol_make.py --filter_size 4 --mod_scheme "QPSK" --gen_seq_len 130000 --data_gen_type 'test' --rand_seed_test 4913
# python symbol_make.py --filter_size 16 --mod_scheme "QPSK" --gen_seq_len 250000 --data_gen_type 'test' --rand_seed_test 4925

python symbol_make.py --total_taps 10 --filter_size 10 --mod_scheme "QPSK" --gen_seq_len 19000 --data_gen_type 'train' --rand_seed_train 9001
python symbol_make.py --total_taps 10 --filter_size 10 --mod_scheme "QPSK" --gen_seq_len 190000 --data_gen_type 'test' --rand_seed_test 4001

python symbol_make.py --total_taps 10 --filter_size 20 --mod_scheme "QPSK" --gen_seq_len 29000 --data_gen_type 'train' --rand_seed_train 9002
python symbol_make.py --total_taps 10 --filter_size 20 --mod_scheme "QPSK" --gen_seq_len 290000 --data_gen_type 'test' --rand_seed_test 4002

python symbol_make.py --total_taps 10 --filter_size 50 --mod_scheme "QPSK" --gen_seq_len 59000 --data_gen_type 'train' --rand_seed_train 9005
python symbol_make.py --total_taps 10 --filter_size 50 --mod_scheme "QPSK" --gen_seq_len 590000 --data_gen_type 'test' --rand_seed_test 4005

python symbol_make.py --total_taps 10 --filter_size 100 --mod_scheme "QPSK" --gen_seq_len 109000 --data_gen_type 'train' --rand_seed_train 9010
python symbol_make.py --total_taps 10 --filter_size 100 --mod_scheme "QPSK" --gen_seq_len 1090000 --data_gen_type 'test' --rand_seed_test 4010

python symbol_make.py --total_taps 10 --filter_size 200 --mod_scheme "QPSK" --gen_seq_len 209000 --data_gen_type 'train' --rand_seed_train 9020
python symbol_make.py --total_taps 10 --filter_size 200 --mod_scheme "QPSK" --gen_seq_len 2090000 --data_gen_type 'test' --rand_seed_test 4020

python symbol_make.py --total_taps 10 --filter_size 500 --mod_scheme "QPSK" --gen_seq_len 509000 --data_gen_type 'train' --rand_seed_train 9050
python symbol_make.py --total_taps 10 --filter_size 500 --mod_scheme "QPSK" --gen_seq_len 5090000 --data_gen_type 'test' --rand_seed_test 4050

python symbol_make.py --total_taps 10 --filter_size 1000 --mod_scheme "QPSK" --gen_seq_len 1009000 --data_gen_type 'train' --rand_seed_train 9100
python symbol_make.py --total_taps 10 --filter_size 1000 --mod_scheme "QPSK" --gen_seq_len 10090000 --data_gen_type 'test' --rand_seed_test 4100