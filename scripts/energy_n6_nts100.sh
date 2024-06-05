#!/user/bin/env bash

conda activate qcopt

cd ../example/StochasticModel/

python energy_ratio.py --name='EnergyRevision' --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=0 \
  --initial_type='CONSTANT' --offset=0.5 --num_scenario=300 --cvar=0.05 --var=0.01 --binary=1 --weight=1.0 --ts_b=200
python energy_ratio.py --name='EnergyRevision' --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=0 \
  --initial_type='CONSTANT' --offset=0.5 --num_scenario=300 --cvar=0.05 --var='[0.01,0.05]' --binary=1 --weight=1.0 --ts_b=200
python energy_ratio.py --name='EnergyRevision' --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=0 \
  --initial_type='CONSTANT' --offset=0.5 --num_scenario=300 --cvar=0.05 --var='[0.01,0.1]' --binary=1 --weight=1.0 --ts_b=200
python energy_ratio.py --name='EnergyRevision' --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=0 \
  --initial_type='CONSTANT' --offset=0.5 --num_scenario=300 --cvar=0.05 --var='[0.05,0.01]' --binary=1 --weight=1.0 --ts_b=200
python energy_ratio.py --name='EnergyRevision' --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=0 \
  --initial_type='CONSTANT' --offset=0.5 --num_scenario=300 --cvar=0.05 --var=0.05 --binary=1 --weight=1.0 --ts_b=200
python energy_ratio.py --name='EnergyRevision' --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=0 \
  --initial_type='CONSTANT' --offset=0.5 --num_scenario=300 --cvar=0.05 --var='[0.05,0.1]' --binary=1 --weight=1.0 --ts_b=200
python energy_ratio.py --name='EnergyRevision' --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=0 \
  --initial_type='CONSTANT' --offset=0.5 --num_scenario=300 --cvar=0.05 --var='[0.1,0.01]' --binary=1 --weight=1.0 --ts_b=200
python energy_ratio.py --name='EnergyRevision' --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=0 \
  --initial_type='CONSTANT' --offset=0.5 --num_scenario=300 --cvar=0.05 --var='[0.1,0.05]' --binary=1 --weight=1.0 --ts_b=200
python energy_ratio.py --name='EnergyRevision' --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=0 \
  --initial_type='CONSTANT' --offset=0.5 --num_scenario=300 --cvar=0.05 --var=0.1 --binary=1 --weight=1.0 --ts_b=200

cd ../OutSampleTest/

python energy_ratio.py --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=1000 --group=10 --num_scenario=500 \
  --var=0.01 --cvar=0.05 --draw=0 --control="../../result/control/Direct/EnergyRevision6_evotime5.0_n_ts50_ptypeCONSTANT_offset0.5_instance1_scenario300_cvar0.05_mean0_var0.01_weight1.0.csv"
python energy_ratio.py --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=1000 --group=10 --num_scenario=500 \
  --var='[0.01,0.05]' --cvar=0.05 --draw=0 --control="../../result/control/Direct/EnergyRevision6_evotime5.0_n_ts50_ptypeCONSTANT_offset0.5_instance1_scenario300_cvar0.05_mean0_var[0.01,0.05]_weight1.0.csv"
python energy_ratio.py --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=1000 --group=10 --num_scenario=500 \
  --var='[0.01,0.1]' --cvar=0.05 --draw=0 --control="../../result/control/Direct/EnergyRevision6_evotime5.0_n_ts50_ptypeCONSTANT_offset0.5_instance1_scenario300_cvar0.05_mean0_var[0.01,0.1]_weight1.0.csv"
python energy_ratio.py --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=1000 --group=10 --num_scenario=500 \
  --var='[0.05,0.01]' --cvar=0.05 --draw=0 --control="../../result/control/Direct/EnergyRevision6_evotime5.0_n_ts50_ptypeCONSTANT_offset0.5_instance1_scenario300_cvar0.05_mean0_var[0.05,0.01]_weight1.0.csv"
python energy_ratio.py --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=1000 --group=10 --num_scenario=500 \
  --var=0.05 --cvar=0.05 --draw=0 --control="../../result/control/Direct/EnergyRevision6_evotime5.0_n_ts50_ptypeCONSTANT_offset0.5_instance1_scenario300_cvar0.05_mean0_var0.05_weight1.0.csv"
python energy_ratio.py --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=1000 --group=10 --num_scenario=500 \
  --var='[0.05,0.1]' --cvar=0.05 --draw=0 --control="../../result/control/Direct/EnergyRevision6_evotime5.0_n_ts50_ptypeCONSTANT_offset0.5_instance1_scenario300_cvar0.05_mean0_var[0.05,0.1]_weight1.0.csv"
python energy_ratio.py --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=1000 --group=10 --num_scenario=500 \
  --var='[0.1,0.01]' --cvar=0.05 --draw=0 --control="../../result/control/Direct/EnergyRevision6_evotime5.0_n_ts50_ptypeCONSTANT_offset0.5_instance1_scenario300_cvar0.05_mean0_var[0.1,0.01]_weight1.0.csv"
python energy_ratio.py --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=1000 --group=10 --num_scenario=500 \
  --var='[0.1,0.05]' --cvar=0.05 --draw=0 --control="../../result/control/Direct/EnergyRevision6_evotime5.0_n_ts50_ptypeCONSTANT_offset0.5_instance1_scenario300_cvar0.05_mean0_var[0.1,0.05]_weight1.0.csv"
python energy_ratio.py --n=6 --num_edges=3 --rgraph=1 --g_seed=1 --evo_time=5 --n_ts=50 --r_seed=1000 --group=10 --num_scenario=500 \
  --var=0.1 --cvar=0.05 --draw=0 --control="../../result/control/Direct/EnergyRevision6_evotime5.0_n_ts50_ptypeCONSTANT_offset0.5_instance1_scenario300_cvar0.05_mean0_var0.1_weight1.0.csv"

cd ../../scripts/
