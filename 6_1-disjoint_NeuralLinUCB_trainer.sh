#!/bin/bash

# Training date	Model checkpoint	validation date
# 2026-02-13	1771067485	2026-02-14
# 2026-02-12	1770979662	2026-02-13
# 2026-02-11	1770893443	2026-02-12
# 2026-02-10	1770809120	2026-02-11
# 2026-02-08	1770636411	2026-02-09
# 2026-02-07	1770548252	2026-02-08
# 2026-02-06	1770463379	2026-02-07
# 2026-02-05	1770375089	2026-02-06
# 2026-02-04	1770289082	2026-02-05
# 2026-02-03	1770203382	2026-02-04
# 2026-02-02	1770115825	2026-02-03
# 2026-02-01	1770031258	2026-02-02
# 2026-02-09	1770723470	2026-02-10

python -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-13 --checkpoint 1771067485 &
python -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-12 --checkpoint 1770979662 &
python -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-11 --checkpoint 1770893443 &
python -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-10 --checkpoint 1770809120 &
python -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-08 --checkpoint 1770636411 &
python -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-07 --checkpoint 1770548252 &
python -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-06 --checkpoint 1770463379 &
python -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-05 --checkpoint 1770375089 &
python -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-04 --checkpoint 1770289082 &
python -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-03 --checkpoint 1770203382 &
python -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-02 --checkpoint 1770115825 &
python -m 6_1-disjoint_NeuralLinUCB_trainer.py tpfy-v3-mtl-r2 2026-02-01 --checkpoint 1770031258 &

wait

echo "All scripts have completed."