import numpy as np
import math, os
from tqdm import tqdm
import random
import csv
from collections import OrderedDict
import json, pickle
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
absolute_path = os.path.abspath(__file__)
DATA_PATH = "/".join(absolute_path.split("/")[:-1]+["datas"])


def preprocess_assay(lines):
    test_sup_num = 16
    x_tmp = []
    smiles_list = []
    activity_list = []

    if lines is None:
        return None

    if len(lines) > 10000:
        return None

    for line in lines:
        smiles = line["smiles"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fingerprints_vect = rdFingerprintGenerator.GetCountFPs(
            [mol], fpType=rdFingerprintGenerator.MorganFP
        )[0]
        fp_numpy = np.zeros((0,), np.int8)  # Generate target pointer to fill
        DataStructs.ConvertToNumpyArray(fingerprints_vect, fp_numpy)
        pic50_exp = line["pic50_exp"]
        activity_list.append(pic50_exp)
        x_tmp.append(fp_numpy)
        smiles_list.append(smiles)

    x_tmp = np.array(x_tmp).astype(np.float32)
    affis = np.array(activity_list).astype(np.float32)
    if len(x_tmp) < 20 and lines[0].get("domain", "none") in ['chembl', 'bdb', 'pqsar', 'fsmol']:
        return None
    return x_tmp, affis, smiles_list


def read_chembl_assay():
    datas = csv.reader(open(f"{DATA_PATH}/chembl/chembl_processed_chembl32.csv", "r"),
                       delimiter=',')
    assay_id_dicts = {}


    # kd_assay_set = set()
    for line in datas:
        unit = line[7]
        if unit=="%":
            continue
        assay_id = "{}_{}_{}".format(line[11], line[7], line[8]).replace("/", "_")
        if assay_id not in assay_id_dicts:
            assay_id_dicts[assay_id] = []
        smiles = line[13]
        assay_type = line[9]
        bao_endpoint = line[4]
        bao_format = line[10]
        std_type = line[8]
        # if std_type.lower() != "kd":
        #     continue
        unit = line[7]
        std_rel = line[5]

        if std_rel != "=":
            continue
        is_does = unit in ['ug.mL-1', 'ug ml-1', 'mg.kg-1', 'mg kg-1',
                           'mg/L', 'ng/ml', 'mg/ml', 'ug kg-1', 'mg/kg/day', 'mg kg-1 day-1',
                           "10'-4 ug/ml", 'M kg-1', "10'-6 ug/ml", 'ng/L', 'pmg kg-1', "10'-8mg/ml",
                           'ng ml-1', "10'-3 ug/ml", "10'-1 ug/ml", ]
        pic50_exp = -math.log10(float(line[6]))
        affi_prefix = line[5]
        ligand_info = {
            "assay_type": std_type,
            "smiles": smiles,
            "pic50_exp": pic50_exp,
            "affi_prefix": affi_prefix,
            "is_does": is_does,
            "chembl_assay_type": assay_type,
            "bao_endpoint": bao_endpoint,
            "bao_format": bao_format,
            "unit": unit,
            "domain": "chembl"
        }
        assay_id_dicts[assay_id].append(ligand_info)
    
    # print(list(kd_assay_set))
    # exit()
    assay_id_dicts_new = {}
    for assay_id, ligands in assay_id_dicts.items():
        pic50_exp_list = [x["pic50_exp"] for x in ligands]
        pic50_std = np.std(pic50_exp_list)
        if pic50_std <= 0.2:
            continue
        if len(ligands) < 20:
            continue
        if len(ligands) >= 128:
          random.seed(1111)
          random.shuffle(ligands)
          ligands = ligands[:128]
        assay_id_dicts_new[assay_id] = ligands

    return assay_id_dicts_new


import torch
class AssayDataset:
    def __init__(self):
        self.datas = read_chembl_assay()

    def get_knn_assays(self, assay_id_list, weight_list):
        ret_data = []
        for aid, ret_weight in zip(assay_id_list, weight_list):
          X, y, smis = preprocess_assay(self.datas[aid])
          X = torch.tensor(X)
          y = torch.tensor(y)
          split = torch.ones_like(y)
          preprocess_assay(self.datas[aid])
          ret_data.append([X, y, split, aid, ret_weight, smis])

        ret_data_flat = []
        for i in range(6):
          ret_data_flat.append([x[i] for x in ret_data])
        return tuple(ret_data_flat)

  
