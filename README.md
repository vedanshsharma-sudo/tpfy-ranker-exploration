# tpfy-ranker-exploration

## Matrices on s3 to be consumed on serving side everyday - 
s3://p13n-reco-offline-models-prod/models/tpfy/tpfy-v3-neural-linucb/latest_matrices/

## Connect with the EC2 instance - 
ssh box2-prod-sg (Follow this document for setting it up - https://docs.google.com/document/d/1JIKqyGR93y4HuLvBuWkoRgHn-CXI1CLIA4mbvUj0Vgg/edit?tab=t.0)

## Virtual env - 
conda activate tpfy_ranker_py37 <br>
Otherwise install requirements.txt <br>

## Training and evaluation - 
$ ./tpfy/neural_linucb_run.sh  <br>

## Training script - 
python3.7 -m tpfy.neural_linucb_trainer tpfy-v3-mtl-r2 2026-02-09 --checkpoint 1770723470 <br>

## Evaluation scripts - 
python3.7 -m tpfy.neural_linucb_evaluator tpfy-v3-mtl-r2 2026-02-09 --checkpoint 1770723470 --validation_run 600 <br>
python3.7 -m tpfy.neural_linucb_get_evaluation_results 2026-02-09 --validation_run 600 <br>

## ETL script for validation dataset creation - 
3_1-create-validation-dataset.py (Run it on databricks cluster) <br>

## Creation of cumulative, reseted, penalized matrices - 
5.1-save-matrices.ipynb <br>

## Creation of popularity tag - 
SQL script to be run on databricks - popularity_tag_creation.txt <br>

## Get the model checkpoints - 
aws s3 ls s3://p13n-reco-offline-models-prod/models/tpfy/tpfy-v3-mtl-r2/ <br>

## Daily training files - 
aws s3 ls s3://p13n-reco-offline-prod/dataset_v5/tpfy-impr-v3/daily-mtl-extracted-cms3/ <br>

## Validation files path - (Created using 3_1-create-validation-dataset.py on databricks)
s3://p13n-reco-offline-prod/upload_objects/test_vedansh/daily-mtl-extracted-cms3-minimum-5-contents/ <br>
