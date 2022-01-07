# python main.py --seq_len 40000 --epochs 30

# python main.py --seq_len 80000 --filter_type "Linear" --filter_size 70 --rand_seed_train 9995 --rand_seed_test 4995
# python main.py --seq_len 80000 --filter_type "NN" --filter_size 18 --epochs 30 --rand_seed_train 9993 --rand_seed_test 4993

# Start simulation

# python main.py --total_taps 10 --filter_size 5 --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 200 --test_seq_len 40000
# python main.py --total_taps 10 --filter_size 5 --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 400 --test_seq_len 40000
# python main.py --total_taps 10 --filter_size 5 --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 1000 --test_seq_len 40000
# python main.py --total_taps 10 --filter_size 5 --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 2000 --test_seq_len 40000
# python main.py --total_taps 10 --filter_size 5 --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 4000 --test_seq_len 40000
# python main.py --total_taps 10 --filter_size 5 --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 8000 --test_seq_len 40000
# python main.py --total_taps 10 --filter_size 5 --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 12000 --test_seq_len 40000
# python main.py --total_taps 10 --filter_size 5 --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 16000 --test_seq_len 40000
# python main.py --total_taps 10 --filter_size 5 --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 20000 --test_seq_len 40000
# python main.py --total_taps 10 --filter_size 5 --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 24000 --test_seq_len 40000
# python main.py --total_taps 10 --filter_size 5 --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 30000 --test_seq_len 40000
# python main.py --total_taps 10 --filter_size 5 --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 36000 --test_seq_len 40000
# python main.py --total_taps 10 --filter_size 5 --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 40000 --test_seq_len 40000

# python main.py --total_taps 10 --filter_size 5 --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 200 --test_seq_len 40000 --bs 16 
# python main.py --total_taps 10 --filter_size 5 --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 400 --test_seq_len 40000 --bs 16 
# python main.py --total_taps 10 --filter_size 5 --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 1000 --test_seq_len 40000 --bs 16 
# python main.py --total_taps 10 --filter_size 5 --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 2000 --test_seq_len 40000 --bs 64 
# python main.py --total_taps 10 --filter_size 5 --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 4000 --test_seq_len 40000 --bs 64 
# python main.py --total_taps 10 --filter_size 5 --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 8000 --test_seq_len 40000 --bs 64 
python main.py --epochs 80 --lr 8e-4 --total_taps 10 --filter_size 24 --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 20000 --test_seq_len 20000 --bs 64 
# python main.py --total_taps 10 --filter_size 24 --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 16000 --test_seq_len 40000 --bs 64
# python main.py --total_taps 10 --filter_size 5 --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 20000 --test_seq_len 40000 --bs 256 
# python main.py --total_taps 10 --filter_size 5 --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 24000 --test_seq_len 40000 --bs 256 
# python main.py --total_taps 10 --filter_size 5 --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 30000 --test_seq_len 40000 --bs 256 
# python main.py --total_taps 10 --filter_size 5 --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 36000 --test_seq_len 40000 --bs 256 
# python main.py --total_taps 10 --filter_size 5 --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 40000 --test_seq_len 40000 --bs 256