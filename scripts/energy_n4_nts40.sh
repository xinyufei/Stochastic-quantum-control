#!/user/bin/env bash

conda activate qcopt

cd ../example/CVaR/
python energy.py --n=4 --num_edges=1 --rgraph=1 --g_seed=1 --evo_time=2 --n_ts=40 --r_seed=0 \
  --initial_type=CONSTANT --offset=0.5 --num_scenario=10 --cvar=0.1 --bound=0.3

cd ../OutSampleTest/
python energy.py --n=4 --num_edges=1 --rgraph=1 --g_seed=1 --evo_time=2 --n_ts=40 --r_seed=1 --num_scenario=100 \
  --bound=0.3 --cvar=0.1 --control="../../control/Average/Energy4_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance1.csv"
python energy.py --n=4 --num_edges=1 --rgraph=1 --g_seed=1 --evo_time=2 --n_ts=40 --r_seed=1 --num_scenario=100 \
  --bound=0.3 --cvar=0.1 --control="../../control/Stepcvar/Energy4_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance1_scenario10_cvar0.1_bound0.3.csv"

cd ../../scripts/
