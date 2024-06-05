#!/user/bin/env bash
cd ../example/StochasticModel/

#python Molecule_stochastic.py --name="MoleculeRevisionlr0p05" --molecule="H2" --qubit_num=2 --evo_time=20 --n_ts=50 --r_seed=0 --gen_target=0 \
#   --target="../../result/target/Molecule_H2_target.csv" --sum_penalty=1 --max_iter=2000 \
#   --initial_type="CONSTANT" --offset=0.2 --num_scenario=20 --cvar=0.05 --var=0.01 --binary=1 --ts_b=4000 --weight=1.0 --lr=0.05 > output_scenario20_revision_lr0p05.txt

cd ../OutSampleTest/
#python Molecule.py --name="MoleculeRevision" --molecule="H2" --qubit_num=2 --evo_time=20 --n_ts=50 --r_seed=1000 --gen_target=0 \
#--target="../../result/target/Molecule_H2_target.csv" --group=10 --num_scenario=500 --cvar=0.05 --var=0.05 --draw=0 \
#--control="../../result/control/Stochastic/revision/MoleculeTestlr0p05Sto_H2_evotime20.0_n_ts50_ptypeCONSTANT_offset0.2_sum_penalty1.0_scenario20_cvar0.05_mean0_var0.05_weight0.5.csv"
#python Molecule.py --name="MoleculeRevision" --molecule="H2" --qubit_num=2 --evo_time=20 --n_ts=50 --r_seed=1000 --gen_target=0 \
#--target="../../result/target/Molecule_H2_target.csv" --group=10 --num_scenario=500 --cvar=0.05 --var=0.01 --draw=0 \
#--control="../../result/control/Stochastic/revision/MoleculeRevisionlr0p05Sto_H2_evotime20.0_n_ts50_ptypeCONSTANT_offset0.2_sum_penalty1.0_scenario20_cvar0.05_mean0_var0.01_weight1.0.csv"
#python Molecule.py --name="MoleculeRevision" --molecule="H2" --qubit_num=2 --evo_time=20 --n_ts=50 --r_seed=1000 --gen_target=0 \
#--target="../../result/target/Molecule_H2_target.csv" --group=10 --num_scenario=500 --cvar=0.05 --var='[0.01,0.05]' --draw=0 \
#--control="../../result/control/Stochastic/revision/MoleculeRevisionlr0p05Sto_H2_evotime20.0_n_ts50_ptypeCONSTANT_offset0.2_sum_penalty1.0_scenario20_cvar0.05_mean0_var[0.01,0.05]_weight1.0.csv"
#python Molecule.py --name="MoleculeRevision" --molecule="H2" --qubit_num=2 --evo_time=20 --n_ts=50 --r_seed=1000 --gen_target=0 \
#--target="../../result/target/Molecule_H2_target.csv" --group=10 --num_scenario=500 --cvar=0.05 --var='[0.05,0.01]' --draw=0 \
#--control="../../result/control/Stochastic/revision/MoleculeRevisionlr0p05Sto_H2_evotime20.0_n_ts50_ptypeCONSTANT_offset0.2_sum_penalty1.0_scenario20_cvar0.05_mean0_var[0.05,0.01]_weight1.0.csv"
#python Molecule.py --name="MoleculeRevision" --molecule="H2" --qubit_num=2 --evo_time=20 --n_ts=50 --r_seed=1000 --gen_target=0 \
#--target="../../result/target/Molecule_H2_target.csv" --group=10 --num_scenario=500 --cvar=0.05 --var=0.05 --draw=0 \
#--control="../../result/control/Stochastic/revision/MoleculeRevisionlr0p05Sto_H2_evotime20.0_n_ts50_ptypeCONSTANT_offset0.2_sum_penalty1.0_scenario20_cvar0.05_mean0_var0.05_weight1.0.csv"

cd ../../scripts/