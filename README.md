Neural_filter
================

# To-do
1. Data construction
- ~~Save channel taps~~
- ~~Generate channel taps with fixed random seed~~
- ~~First, implement without noise (SNR should be considered later)~~

2. LMMSE equalizer
- ~~Optimal filter~~
- ~~Implemented only by matrix calculation~~

Requirement (environment)
-----------------
Refer to [requirements.txt](./requirements.txt)


Usage (run)
----------------
__0. Make a dataset__

Move to `data` and check the arguments option:
```bash
$ python symbol_make.py --help
```

Execute following bash file:
```bash
$ bash generate_data.sh
```

__1. Execute an equalizer__

Execute following bash file:
```bash
$ bash execute_main.sh
```