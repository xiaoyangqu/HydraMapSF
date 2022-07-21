# HydraMapSF
HydraMap-based features generation


CONTENTS

Notebooks

    1. Generating hydration sites by Hydration sites analysis:
    sh gen_wat4.sh ./data
    
    2. Generaing HydraMap-based features, RF-score features or Extended Connectivity Interaction Features:
    python calFeature.py

    3. Train and test:
    python train_model.py --train feature_trainset.csv --test feature_testset.csv --model xgb --out result.txt

    4. save SF model：
    python train_model.py --train feature_trainset.csv --savemod --model xgb

    5. run SF trained model:
    python train_model.py --test feature_testset.csv --runmod --out result.txt

   
Folders
    
    Data: protein & ligand structure files, experimental binding affinity file, and PDBID list file
    
    
Notes:

    1. Protein PDB files are assumed to contain coordinates for all heavy atoms
