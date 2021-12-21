# python main.py --seq_len 40000 --epochs 30

# python main.py --seq_len 80000 --filter_type "Linear" --filter_size 70 --rand_seed_train 9995 --rand_seed_test 4995
# python main.py --seq_len 80000 --filter_type "NN" --filter_size 18 --epochs 30 --rand_seed_train 9993 --rand_seed_test 4993

#python main.py --seq_len 80000 --filter_type "Linear" --filter_size 70 --rand_seed_train 9995 --rand_seed_test 4995
#python main.py --seq_len 80000 --filter_type "NN" --filter_size 70 --epochs 30 --rand_seed_train 9995 --rand_seed_test 4995 

#python main.py --seq_len 80000 --filter_type "Linear" --filter_size 1000 --rand_seed_train 6999 --rand_seed_test 1999 --total_taps 8
#python main.py --seq_len 80000 --filter_type "NN" --filter_size 500 --epochs 15 --rand_seed_train 8999 --rand_seed_test 3999

# Start simulation

# python main.py --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 200 --test_seq_len 200
# python main.py --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 400 --test_seq_len 400
# python main.py --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 1000 --test_seq_len 1000
# python main.py --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 2000 --test_seq_len 2000
# python main.py --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 4000 --test_seq_len 4000
python main.py --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 200 --test_seq_len 40000
python main.py --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 16000 --test_seq_len 40000
# python main.py --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 20000 --test_seq_len 20000
# python main.py --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 24000 --test_seq_len 24000
# python main.py --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 30000 --test_seq_len 30000
# python main.py --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 36000 --test_seq_len 36000
# python main.py --filter_type "Linear" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 40000 --test_seq_len 40000

# python main.py --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 200 --test_seq_len 200 --bs 16 --epochs 40
# python main.py --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 400 --test_seq_len 400 --bs 16 --epochs 40
# python main.py --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 1000 --test_seq_len 1000 --bs 16 --epochs 40
# python main.py --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 2000 --test_seq_len 2000 --bs 64 --epochs 40
# python main.py --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 4000 --test_seq_len 4000 --bs 64 --epochs 40
# python main.py --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 8000 --test_seq_len 8000 --bs 64 --epochs 40
# python main.py --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 16000 --test_seq_len 16000 --bs 64 --epochs 40
# python main.py --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 20000 --test_seq_len 20000 --bs 256 --epochs 40
# python main.py --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 24000 --test_seq_len 24000 --bs 256 --epochs 40
# python main.py --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 30000 --test_seq_len 30000 --bs 256 --epochs 40
# python main.py --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 36000 --test_seq_len 36000 --bs 256 --epochs 40
# python main.py --filter_type "NN" --rand_seed_train 9997 --rand_seed_test 4997 --train_seq_len 40000 --test_seq_len 40000 --bs 256 --epochs 40