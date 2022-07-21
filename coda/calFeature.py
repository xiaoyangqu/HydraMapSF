#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial.distance import cdist
from itertools import product
import os

import time
from pymol import cmd



# In[2]:

# ### For ECIF
# Possible predefined protein atoms
ECIF_ProteinAtoms = ['C;4;1;3;0;0', 'C;4;2;1;1;1', 'C;4;2;2;0;0', 'C;4;2;2;0;1',
                     'C;4;3;0;0;0', 'C;4;3;0;1;1', 'C;4;3;1;0;0', 'C;4;3;1;0;1',
                     'C;5;3;0;0;0', 'C;6;3;0;0;0', 'N;3;1;2;0;0', 'N;3;2;0;1;1',
                     'N;3;2;1;0;0', 'N;3;2;1;1;1', 'N;3;3;0;0;1', 'N;4;1;2;0;0',
                     'N;4;1;3;0;0', 'N;4;2;1;0;0', 'O;2;1;0;0;0', 'O;2;1;1;0;0',
                     'S;2;1;1;0;0', 'S;2;2;0;0;0']

# Possible ligand atoms according to the PDBbind 2016 "refined set"
ECIF_LigandAtoms = ['Br;1;1;0;0;0', 'C;3;3;0;1;1', 'C;4;1;1;0;0', 'C;4;1;2;0;0',
                     'C;4;1;3;0;0', 'C;4;2;0;0;0', 'C;4;2;1;0;0', 'C;4;2;1;0;1',
                     'C;4;2;1;1;1', 'C;4;2;2;0;0', 'C;4;2;2;0;1', 'C;4;3;0;0;0',
                     'C;4;3;0;0;1', 'C;4;3;0;1;1', 'C;4;3;1;0;0', 'C;4;3;1;0;1',
                     'C;4;4;0;0;0', 'C;4;4;0;0;1', 'C;5;3;0;0;0', 'C;5;3;0;1;1',
                     'C;6;3;0;0;0', 'Cl;1;1;0;0;0', 'F;1;1;0;0;0', 'I;1;1;0;0;0',
                     'N;3;1;0;0;0', 'N;3;1;1;0;0', 'N;3;1;2;0;0', 'N;3;2;0;0;0',
                     'N;3;2;0;0;1', 'N;3;2;0;1;1', 'N;3;2;1;0;0', 'N;3;2;1;0;1',
                     'N;3;2;1;1;1', 'N;3;3;0;0;0', 'N;3;3;0;0;1', 'N;3;3;0;1;1',
                     'N;4;1;2;0;0', 'N;4;1;3;0;0', 'N;4;2;1;0;0', 'N;4;2;2;0;0',
                     'N;4;2;2;0;1', 'N;4;3;0;0;0', 'N;4;3;0;0;1', 'N;4;3;1;0;0',
                     'N;4;3;1;0;1', 'N;4;4;0;0;0', 'N;4;4;0;0;1', 'N;5;2;0;0;0',
                     'N;5;3;0;0;0', 'N;5;3;0;1;1', 'O;2;1;0;0;0', 'O;2;1;1;0;0',
                     'O;2;2;0;0;0', 'O;2;2;0;0;1', 'O;2;2;0;1;1', 'P;5;4;0;0;0',
                     'P;6;4;0;0;0', 'P;6;4;0;0;1', 'P;7;4;0;0;0', 'S;2;1;0;0;0',
                     'S;2;1;1;0;0', 'S;2;2;0;0;0', 'S;2;2;0;0;1', 'S;2;2;0;1;1',
                     'S;3;3;0;0;0', 'S;3;3;0;0;1', 'S;4;3;0;0;0', 'S;6;4;0;0;0',
                     'S;6;4;0;0;1', 'S;7;4;0;0;0']

ECIF_WaterAtoms = ['Ot;2;0;2;0;0','Ol;2;0;2;0;0','Ow;2;0;2;0;0','Oa;2;0;2;0;0',
                    'Op;2;0;2;0;0','On;2;0;2;0;0']

PossibleECIF = [i[0]+"-"+i[1] for i in product(ECIF_ProteinAtoms, ECIF_LigandAtoms)]
PossibleECIF2 = [i[0]+"-"+i[1] for i in product(ECIF_WaterAtoms, ECIF_ProteinAtoms)]
PossibleECIF3 = [i[0]+"-"+i[1] for i in product(ECIF_WaterAtoms, ECIF_LigandAtoms)]

# ### For RF-Score


ELEMENTS_ProteinAtoms = ["C","N","O", "S"]
ELEMENTS_LigandAtoms = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]
ELEMENTS_WaterAtoms = ["Ot","Ol","Ow","Oa","Op","On"]
PossibleELEMENTS = [i[0]+"-"+i[1] for i in product(ELEMENTS_ProteinAtoms, ELEMENTS_LigandAtoms)]
PossibleELEMENTS2 = [i[0]+"-"+i[1] for i in product(ELEMENTS_WaterAtoms, ELEMENTS_ProteinAtoms)]
PossibleELEMENTS3 = [i[0]+"-"+i[1] for i in product(ELEMENTS_WaterAtoms, ELEMENTS_LigandAtoms)]


def GetAtomType(atom):
# This function takes an atom in a molecule and returns its type as defined for ECIF
    
    AtomType = [atom.GetSymbol(),
                str(atom.GetExplicitValence()),
                str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"])),
                str(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"])),
                str(int(atom.GetIsAromatic())),
                str(int(atom.IsInRing())), 
               ]

    return(";".join(AtomType))


# ### Ligands are loaded from an SDF file in a dataframe format considering the atom type definitions



def LoadSDFasDF(SDF):
# This function takes an SDF for a ligand as input and returns it as a pandas DataFrame with its atom types labeled according to ECIF
    
    m = Chem.MolFromMolFile(SDF, sanitize=False)
    m.UpdatePropertyCache(strict=False)
    
    ECIF_atoms = []

    for atom in m.GetAtoms():
        if atom.GetSymbol() != "H": # Include only non-hydrogen atoms
            entry = [int(atom.GetIdx())]
            entry.append(GetAtomType(atom))
            pos = m.GetConformer().GetAtomPosition(atom.GetIdx())
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            ECIF_atoms.append(entry)

    df = pd.DataFrame(ECIF_atoms)
    df.columns = ["ATOM_INDEX", "ECIF_ATOM_TYPE","X","Y","Z"]
    if len(set(df["ECIF_ATOM_TYPE"]) - set(ECIF_LigandAtoms)) > 0:
        print("WARNING: Ligand contains unsupported atom types. Only supported atom-type pairs are counted.")
    return(df)



Atom_Keys=pd.read_csv("PDB_Atom_Keys.csv", sep=",")
def LoadPDBasDF(PDB):
# This function takes a PDB for a protein as input and returns it as a pandas DataFrame with its atom types labeled according to ECIF

    ECIF_atoms = []
    
    f = open(PDB)
    for i in f:
        if i[:4] == "ATOM":
            # Include only non-hydrogen atoms
            if (len(i[12:16].replace(" ","")) < 4 and i[12:16].replace(" ","")[0] != "H") or (len(i[12:16].replace(" ","")) == 4 and i[12:16].replace(" ","")[1] != "H" and i[12:16].replace(" ","")[0] != "H"):
                ECIF_atoms.append([int(i[6:11]),
                         i[17:20]+"-"+i[12:16].replace(" ",""),
                         float(i[30:38]),
                         float(i[38:46]),
                         float(i[46:54])
                        ])
                
    f.close()
    
    df = pd.DataFrame(ECIF_atoms, columns=["ATOM_INDEX","PDB_ATOM","X","Y","Z"])
    df = df.merge(Atom_Keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[["ATOM_INDEX", "ECIF_ATOM_TYPE", "X", "Y", "Z"]].sort_values(by="ATOM_INDEX").reset_index(drop=True)
    if list(df["ECIF_ATOM_TYPE"].isna()).count(True) > 0:
        print("WARNING: Protein contains unsupported atom types. Only supported atom-type pairs are counted.")
    return(df)


#%%

def LoadWatinEnv(protein_f,wat_f,lig_f):
    cmd.delete("all")
    cmd.load(protein_f)
    cmd.load(wat_f)
    cmd.load(lig_f)
    pro=f"{os.path.splitext(os.path.basename(protein_f))[0]}"
    wat=f"{os.path.splitext(os.path.basename(wat_f))[0]}"
    lig=f"{os.path.splitext(os.path.basename(lig_f))[0]}"

    cmd.create('poc', f'br. {pro} w. 4 of {wat}')
    cmd.create('hydrophobes', f'br. {wat} w. 4 of (resn ala+gly+val+ile+leu+phe+met) in poc \
                & not br. {wat} w. 2 of (resn ala+gly+val+ile+leu+phe+met) in poc')
    cmd.create('hydrophilics', f'br. {wat} w. 3.5 of e. n+o+s in (resn arg+lys+his+glu+asp+asn+gln+thr+ser+cys) in poc \
                & not br. {wat} w. 2 of e. n+o+s in (resn arg+lys+his+glu+asp+asn+gln+thr+ser+cys) in poc\
                & br. {wat} w. 3.5 of e. n+o+s+p+f+cl+br+I in {lig}\
                & not br. {wat} w. 2 of e. n+o+s+p+f+cl+br+I in {lig}')
    cmd.create('aromatics', f'br. {wat} w. 4 of (resn phe+tyr+trp+his) in poc \
                & not br. {wat} w. 2 of (resn phe+tyr+trp+his) in poc')
    cmd.create('pos', f'br. {wat} w. 3.5 of e. n+o+s in (resn arg+lys+his) in poc \
                & not br. {wat} w. 2 of e. n+o+s in (resn arg+lys+his) in poc\
                & br. {wat} w. 3.5 of e. n+o+s+p+f+cl+br+I in {lig}\
                & not br. {wat} w. 2 of e. n+o+s+p+f+cl+br+I in {lig}')
    cmd.create('neg', f'br. {wat} w. 3.5 of e. n+o+s in (resn asp+glu) in poc \
                & not br. {wat} w. 2 of e. n+o+s in (resn asp+glu) in poc\
                & br. {wat} w. 3.5 of e. n+o+s+p+f+cl+br+I in {lig}\
                & not br. {wat} w. 2 of e. n+o+s+p+f+cl+br+I in {lig}')

    O_in_pocket = []
    O_in_hydrophobes = []
    O_in_hydrophilics = []
    O_in_aromatics = []
    O_in_pos = []
    O_in_neg = []
    for O in range(1,cmd.count_atoms(wat)+1):
        try:
            pocket = cmd.centerofmass(f'resi {O} in {wat}')
            O_in_pocket.append([str(O)]+['Ot;2;0;2;0;0']+pocket)
        except:
            pass
        try:
            hydrophobes = cmd.centerofmass(f'resi {O} in hydrophobes')
            O_in_hydrophobes.append([str(O)]+['Ol;2;0;2;0;0']+hydrophobes)
        except:
            pass
        try:
            hydrophilics = cmd.centerofmass(f'resi {O} in hydrophilics')
            O_in_hydrophilics.append([str(O)]+['Ow;2;0;2;0;0']+hydrophilics)
        except:
            pass
        try:    
            aromatics = cmd.centerofmass(f'resi {O} in aromatics')
            O_in_aromatics.append([str(O)]+['Oa;2;0;2;0;0']+aromatics)
        except:
            pass
        try:
            pos = cmd.centerofmass(f'resi {O} in pos')
            O_in_pos.append([str(O)]+['Op;2;0;2;0;0']+pos)
        except:
            pass
        try:    
            neg = cmd.centerofmass(f'resi {O} in neg')
            O_in_neg.append([str(O)]+['On;2;0;2;0;0']+neg)
        except:
            pass
    df_pocket = pd.DataFrame(O_in_pocket, columns=["ATOM_INDEX","ECIF_ATOM_TYPE","X","Y","Z"])
    df_hydrophobes = pd.DataFrame(O_in_hydrophobes, columns=["ATOM_INDEX","ECIF_ATOM_TYPE","X","Y","Z"])
    df_hydrophilics = pd.DataFrame(O_in_hydrophilics, columns=["ATOM_INDEX","ECIF_ATOM_TYPE","X","Y","Z"])
    df_aromatics = pd.DataFrame(O_in_aromatics, columns=["ATOM_INDEX","ECIF_ATOM_TYPE","X","Y","Z"])
    df_pos = pd.DataFrame(O_in_pos, columns=["ATOM_INDEX","ECIF_ATOM_TYPE","X","Y","Z"])
    df_neg = pd.DataFrame(O_in_neg, columns=["ATOM_INDEX","ECIF_ATOM_TYPE","X","Y","Z"])

    df_WatEnv = pd.concat([df_pocket,df_hydrophobes,df_hydrophilics,df_aromatics,df_pos,df_neg],ignore_index = True)

    return df_WatEnv
#%%
def GetWatPairs1(PDB_protein, SDF_ligand, PDB_water,distance_cutoff=11.5):
# This function returns the protein-ligand atom-type pairs for a given distance cutoff
    
    # Load both structures as pandas DataFrames
    Target = LoadPDBasDF(PDB_protein)
    Ligand = LoadSDFasDF(SDF_ligand)
    Water = LoadWatinEnv(PDB_protein ,PDB_water,SDF_ligand)
    for i in ["X","Y","Z"]:
        Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]
    
    # Get all possible pairs
    Pairs = list(product(Target["ECIF_ATOM_TYPE"], Ligand["ECIF_ATOM_TYPE"]))
    Pairs = [x[0]+"-"+x[1] for x in Pairs]
    Pairs = pd.DataFrame(Pairs, columns=["ECIF_PAIR"])
    Distances = cdist(Target[["X","Y","Z"]], Ligand[["X","Y","Z"]], metric="euclidean")
    Distances = Distances.reshape(Distances.shape[0]*Distances.shape[1],1)
    Distances = pd.DataFrame(Distances, columns=["DISTANCE"])
    Pairs1 = pd.concat([Pairs,Distances], axis=1)
    Pairs1 = Pairs1[Pairs1["DISTANCE"] <= distance_cutoff].reset_index(drop=True)

    Pairs1["ELEMENTS_PAIR"] = [x.split("-")[0].split(";")[0]+"-"+x.split("-")[1].split(";")[0] for x in Pairs1["ECIF_PAIR"]]
    return Pairs1  
    # Get W-P pairs
def GetWatPairs2(PDB_protein, SDF_ligand, PDB_water,distance_cutoff=11.5):    
        # Load both structures as pandas DataFrames
    Target = LoadPDBasDF(PDB_protein)
    Ligand = LoadSDFasDF(SDF_ligand)
    Water = LoadWatinEnv(PDB_protein ,PDB_water,SDF_ligand)
    # Take all atoms from the target within a cubic box around the ligand considering the "distance_cutoff criterion"
    for i in ["X","Y","Z"]:
        Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]
    
    Pairs = list(product(Water["ECIF_ATOM_TYPE"], Target["ECIF_ATOM_TYPE"]))
    Pairs = [x[0]+"-"+x[1] for x in Pairs]
    Pairs = pd.DataFrame(Pairs, columns=["ECIF_PAIR"])
    Distances = cdist(Water[["X","Y","Z"]], Target[["X","Y","Z"]], metric="euclidean")
    Distances = Distances.reshape(Distances.shape[0]*Distances.shape[1],1)
    Distances = pd.DataFrame(Distances, columns=["DISTANCE"])
    Pairs2 = pd.concat([Pairs,Distances], axis=1)
    Pairs2 = Pairs2[Pairs2["DISTANCE"] <= distance_cutoff].reset_index(drop=True)
    # Pairs from ELEMENTS could be easily obtained froms pairs from ECIF
    Pairs2["ELEMENTS_PAIR"] = [x.split("-")[0].split(";")[0]+"-"+x.split("-")[1].split(";")[0] for x in Pairs2["ECIF_PAIR"]]
    return Pairs2
    
     # Get W-L pairs
def GetWatPairs3(PDB_protein, SDF_ligand, PDB_water,distance_cutoff=11.5):    
        # Load both structures as pandas DataFrames
    Target = LoadPDBasDF(PDB_protein)
    Ligand = LoadSDFasDF(SDF_ligand)
    Water = LoadWatinEnv(PDB_protein ,PDB_water,SDF_ligand)
    
    for i in ["X","Y","Z"]:
        Water = Water[Water[i] < float(Ligand[i].max())+distance_cutoff]
        Water = Water[Water[i] > float(Ligand[i].min())-distance_cutoff]
    '''
    for i in ["X","Y","Z"]:
        Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]
    '''
    Pairs = list(product(Water["ECIF_ATOM_TYPE"], Ligand["ECIF_ATOM_TYPE"]))
    Pairs = [x[0]+"-"+x[1] for x in Pairs]
    Pairs = pd.DataFrame(Pairs, columns=["ECIF_PAIR"])
    Distances = cdist(Water[["X","Y","Z"]], Ligand[["X","Y","Z"]], metric="euclidean")
    Distances = Distances.reshape(Distances.shape[0]*Distances.shape[1],1)
    Distances = pd.DataFrame(Distances, columns=["DISTANCE"])

    Pairs3 = pd.concat([Pairs,Distances], axis=1)
    Pairs3 = Pairs3[Pairs3["DISTANCE"] <= distance_cutoff].reset_index(drop=True)

    Pairs3["ELEMENTS_PAIR"] = [x.split("-")[0].split(";")[0]+"-"+x.split("-")[1].split(";")[0] for x in Pairs3["ECIF_PAIR"]]
    return Pairs3 

def Get_Wat(PDB_protein, SDF_ligand, PDB_water,distance_cutoff=11.5):

    Pairs2 = GetWatPairs2(PDB_protein, SDF_ligand, PDB_water,distance_cutoff=distance_cutoff)
    ELEMENTS2 = [list(Pairs2["ELEMENTS_PAIR"]).count(x) for x in PossibleELEMENTS2] #w-p
    Pairs3 = GetWatPairs3(PDB_protein, SDF_ligand, PDB_water,distance_cutoff=distance_cutoff)
    ELEMENTS3 = [list(Pairs3["ELEMENTS_PAIR"]).count(x) for x in PossibleELEMENTS3] #w-l
    
    ELEMENTS_Wat = ELEMENTS2 + ELEMENTS3
    return ELEMENTS_Wat

def GetELEMENTS_Wat(PDB_protein, SDF_ligand, PDB_water,distance_cutoff=11.5):

    Pairs1 = GetWatPairs1(PDB_protein, SDF_ligand, PDB_water,distance_cutoff=distance_cutoff)
    ELEMENTS = [list(Pairs1["ELEMENTS_PAIR"]).count(x) for x in PossibleELEMENTS] #p-l
    Pairs2 = GetWatPairs2(PDB_protein, SDF_ligand, PDB_water,distance_cutoff=distance_cutoff)
    ELEMENTS2 = [list(Pairs2["ELEMENTS_PAIR"]).count(x) for x in PossibleELEMENTS2] #w-p
    Pairs3 = GetWatPairs3(PDB_protein, SDF_ligand, PDB_water,distance_cutoff=distance_cutoff)
    ELEMENTS3 = [list(Pairs3["ELEMENTS_PAIR"]).count(x) for x in PossibleELEMENTS3] #w-l 

    ELEMENTS_Wat = ELEMENTS + ELEMENTS2 + ELEMENTS3
    return ELEMENTS_Wat


def GetECIF_ELEMENTSWat(PDB_protein, SDF_ligand, PDB_water,distance_cutoff=11.5):
# Function for the calculation of ELEMENTS

    Pairs1 = GetWatPairs1(PDB_protein, SDF_ligand, PDB_water, distance_cutoff=distance_cutoff)
    ELEMENTS = [list(Pairs1["ECIF_PAIR"]).count(x) for x in PossibleECIF]
    Pairs2 = GetWatPairs2(PDB_protein, SDF_ligand, PDB_water,distance_cutoff=distance_cutoff)
    ELEMENTS2 = [list(Pairs2["ELEMENTS_PAIR"]).count(x) for x in PossibleELEMENTS2] #w-p
    Pairs3 = GetWatPairs3(PDB_protein, SDF_ligand, PDB_water,distance_cutoff=distance_cutoff)
    ELEMENTS3 = [list(Pairs3["ELEMENTS_PAIR"]).count(x) for x in PossibleELEMENTS3] #w-l
    

    ELEMENTS_Wat = ELEMENTS + ELEMENTS2 + ELEMENTS3
    return ELEMENTS_Wat

def embedding_ratio(protein_f,lig_f):
    cmd.delete("all")
    cmd.load(protein_f)
    cmd.load(lig_f)
    pro = f"{os.path.splitext(os.path.basename(protein_f))[0]}"
    lig = f"{os.path.splitext(os.path.basename(lig_f))[0]}"

    cmd.create('complex', f'{pro} or {lig}')

    #cmd.h_add()
    cmd.flag('ignore', 'none')
    cmd.set('dot_solvent', '1')
    cmd.set('dot_density', '3')
    pro_area = cmd.get_area(f"{pro}")
    lig_area = cmd.get_area(f"{lig}")
    complex_area = cmd.get_area("complex")
    
    dsasa = pro_area + lig_area - complex_area
    r = dsasa*0.5/lig_area
    return r


'''
For Fingerprint Pattern
'''

three_one_letter ={'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y', \
    'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A', \
    'GLY':'G', 'PRO':'P', 'CYS':'C'}
three_letter_lower = [ k.lower() for k,v in three_one_letter.items()]
three_letter_lower.sort()
three_letter = list(three_one_letter.keys())
three_letter.sort()
three_letter_headcap = [k.title() for k in three_letter]

def get_residue_occ(protein_f,wat_f):

    cmd.delete("all")
    cmd.load(protein_f)
    cmd.load(wat_f)
    pro=f"{os.path.splitext(os.path.basename(protein_f))[0]}"
    wat=f"{os.path.splitext(os.path.basename(wat_f))[0]}"
    cmd.create('poc', f'br. {pro} w. 4 of {wat}')

    myspace = {'myfunc': []}
    cmd.iterate('poc', 'myfunc.append((resi,resn))', space = myspace)
    tmp = list(set(myspace['myfunc']))

    resi_poc=[tmp[i][1] for i in range(len(tmp))]

    occ = [resi_poc.count(x) for x in three_letter]
    return occ

def get_residue_wat_interact(protein_f,wat_f):
    cmd.delete("all")
    cmd.load(protein_f)
    cmd.load(wat_f)
    pro=f"{os.path.splitext(os.path.basename(protein_f))[0]}"
    wat=f"{os.path.splitext(os.path.basename(wat_f))[0]}"
    cmd.create('poc', f'br. {pro} w. 4 of {wat}')

    dict_count_wat={}
    for resi in three_letter:       
        cmd.create(f'{resi}_wat', f'br. {wat} w. 4 of resn {resi} in poc')
        count_wat = cmd.count_atoms(f'{resi}_wat')
        if count_wat != 0:
            dict_count_wat[resi] = count_wat
        else:
            dict_count_wat[resi] = 0
    resi_wat = list(dict_count_wat.values())
    return resi_wat

def get_buried_ratio(protein_f,lig_f,wat_f):

    cmd.delete("all")
    cmd.load(protein_f)
    cmd.load(lig_f)
    cmd.load(wat_f)
    pro=f"{os.path.splitext(os.path.basename(protein_f))[0]}"
    lig = f"{os.path.splitext(os.path.basename(lig_f))[0]}"
    wat=f"{os.path.splitext(os.path.basename(wat_f))[0]}"
    cmd.create('poc', f'br. {pro} w. 4 of {wat}')
    cmd.flag('ignore', 'none')
    cmd.set('dot_solvent', '0')
    cmd.set('dot_density', '2')
    lig_area = cmd.get_area(f"{lig}") 
    bsasa = []
    for resi in three_letter:       
        #cmd.create(f'{resi}_wat', f'br. {wat} w. 4 of resn {resi} in poc')
        cmd.create(f'{resi}inpoc', f'resn {resi} in poc')
        cmd.create(f'com_{resi}_lig', f'{resi}inpoc or {lig}')
        pro_area = cmd.get_area(f"{resi}inpoc")
        complex_area = cmd.get_area(f"com_{resi}_lig")

        dsasa = pro_area + lig_area - complex_area
        s = round(dsasa*0.5+0,1)
        #r = dsasa*0.5/pro_area
        bsasa.append(s)
    return bsasa

def main(pdbbind_dir,cutoff=12.0,embedding=False,dist=2.0,feat='hydra'):

    data_set_file='list_pdbid.txt'
    exp_file = 'list_exp_2016.csv'
    dlabel = pd.read_csv(f"{pdbbind_dir}/{exp_file}")

    with open(f"{pdbbind_dir}/{data_set_file}") as f:
        data_set = [line.strip() for line in f]

    dfeature=pd.DataFrame()
    #df2=pd.DataFrame()
    for pdb in data_set:
        Protein = f'{pdbbind_dir}/{pdb}_protein.pdb'
        Ligand = f'{pdbbind_dir}/{pdb}_ligand.sdf'
        Water = f'{pdbbind_dir}/{pdb}_wat4.pdb'
        if embedding == "1":
            BR = round(embedding_ratio(Protein, Ligand),2)
            dist = 0.1
            cut = round(BR * cutoff,1)
        elif embedding == "2":
            BR = round(embedding_ratio(Protein, Ligand),2)
            if BR <= 0.5:
                cut = cutoff
            if 0.5 < BR <= 0.75:
                cut = cutoff + dist
            if 0.75 < BR:
                cut = cutoff + 2 * dist 
        else:
            dist = 0.0
            cut = cutoff

        if feat == 'hydra':
            feature = Get_Wat(Protein, Ligand, Water,distance_cutoff=cut)
            feat_index = PossibleELEMENTS2+PossibleELEMENTS3+[f'FP{i}' for i in range(1,61)]
        elif feat == 'rf-score_hydra':
            feature = GetELEMENTS_Wat(Protein, Ligand, Water,distance_cutoff=cut)
            feat_index = PossibleELEMENTS+PossibleELEMENTS2+PossibleELEMENTS3+[f'FP{i}' for i in range(1,61)]
        elif feat == 'ecif_hydra':
            feature = GetECIF_ELEMENTSWat(Protein, Ligand, Water,distance_cutoff=cut)
            feat_index = PossibleECIF+PossibleELEMENTS2+PossibleELEMENTS3+[f'FP{i}' for i in range(1,61)]

        feature1 = get_residue_occ(Protein,Water)
        feature2 = get_residue_wat_interact(Protein,Water)
        feature3 = get_buried_ratio(Protein, Ligand, Water)
        features = feature + feature1 + feature2 + feature3
        df_new = pd.DataFrame(
            [[str(pdb)]+features],
            columns=["pdbid"]+feat_index
        )
        dfeature=dfeature.append(df_new,ignore_index=True)
        dfeature_label = pd.merge(dfeature,dlabel,how="inner", on="pdbid")

    dfeature_label.to_csv(f'{feat}_{cutoff}_{dist}.csv',index='')


  

if __name__=='__main__':

    test_dir="../data"
    main(test_dir,cutoff=15.0,embedding='2',dist=1.0,feat='ecif_hydra')

