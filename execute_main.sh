# python main.py --seq_len 40000 --epochs 30
python main.py --seq_len 80000 --filter_type "Linear" --filter_size 50 --rand_seed_train 9997 --rand_seed_test 4997
python main.py --seq_len 80000 --filter_type "Linear" --filter_size 70 --rand_seed_train 9995 --rand_seed_test 4995
# python main.py --seq_len 80000 --filter_type "NN" --filter_size 18 --epochs 30 --rand_seed_train 9993 --rand_seed_test 4993

#python main.py --seq_len 80000 --filter_type "Linear" --filter_size 70 --rand_seed_train 9995 --rand_seed_test 4995
#python main.py --seq_len 80000 --filter_type "NN" --filter_size 70 --epochs 30 --rand_seed_train 9995 --rand_seed_test 4995 

#python main.py --seq_len 80000 --filter_type "Linear" --filter_size 1000 --rand_seed_train 6999 --rand_seed_test 1999 --total_taps 8
#python main.py --seq_len 80000 --filter_type "NN" --filter_size 500 --epochs 15 --rand_seed_train 8999 --rand_seed_test 3999