# HydraMapSF
HydraMap-based features generation


CONTENTS

Notebooks

    1. Generating hydration sites by Hydration sites analysis:
    sh gen_wat4.sh ./data
    
    2. Generaing HydraMap-based features, RF-score features or Extended Connectivity Interaction Features:
    python calFeature.py

    3. Train and test:
    python train_model.py --train feature_train.csv --test feature_test.csv --model xgb --out result.txt

    4. save SF model：
    python train_model.py --train feature_train.csv --savemod --model xgb

    5. run SF trained model:
    python train_model.py --test feature_test.csv --runmod --out result.txt

   
Folders
    
    data: protein & ligand structure files, experimental binding affinity file, and PDBID list file
    script: scripts related to features generation and modeling   
    
Notes:

    1. Protein PDB files are assumed to contain coordinates for all heavy atoms
