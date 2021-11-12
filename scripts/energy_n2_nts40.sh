#!/user/bin/env bash

conda activate qcopt

cd ../example/CVaR/
python energy.py --n=2 --num_edges=1 --rgraph=0 --evo_time=2 --n_ts=40 --r_seed=0 \
  --initial_type=CONSTANT --offset=0.5 --num_scenario=10 --cvar=0.01 --bound=0.3

cd ../OutSampleTest/
python energy.py --n=2 --num_edges=1 --rgraph=0 --evo_time=2 --n_ts=40 --r_seed=0 --num_scenario=100 \
   --bound=0.3 --control="../../control/Average/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance0.csv"
python energy.py --n=2 --num_edges=1 --rgraph=0 --evo_time=2 --n_ts=40 --r_seed=0 --num_scenario=100 \
   --bound=0.3 --control="../../control/Stepcvar/Energy2_evotime2.0_n_ts40_ptypeCONSTANT_offset0.5_instance0_scenario10_cvar0.01_bound0.3.csv"

cd ../../scripts/