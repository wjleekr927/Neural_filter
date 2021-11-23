python symbol_make.py --mod_scheme "QPSK" --seq_len 6000 --data_gen_type 'train' --filter_size 30 --rand_seed_train 9990 --total_taps 30
python symbol_make.py --mod_scheme "QPSK" --seq_len 6000 --data_gen_type 'test' --filter_size 30 --rand_seed_test 4990 --total_taps 30

#python symbol_make.py --mod_scheme "QPSK" --seq_len 80000 --data_gen_type 'train' --filter_size 70 --rand_seed_train 9995
#python symbol_make.py --mod_scheme "QPSK" --seq_len 80000 --data_gen_type 'test' --filter_size 70 --rand_seed_test 4995

#python symbol_make.py --mod_scheme "QPSK" --seq_len 80000 --data_gen_type 'train' --filter_size 100 --rand_seed_train 8999
#python symbol_make.py --mod_scheme "QPSK" --seq_len 80000 --data_gen_type 'test' --filter_size 100 --rand_seed_test 3999