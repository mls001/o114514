import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import cclib


# 解析高斯计算结果
def parse_gaussian_output(logfile):
    data = cclib.io.ccread(logfile)
    return {
        'homo': data.homos[0],
        'lumo': data.homos[0] + 1,
        'energy': data.scfenergies[-1]
    }


# 生成分子特征数据集
def build_dataset(structures_dir, logfiles_dir):
    dataset = []

    for mol_file in os.listdir(structures_dir):
        # 转换分子结构为SMILES
        mol = Chem.MolFromMolFile(f"{structures_dir}/{mol_file}")
        smiles = Chem.MolToSmiles(mol)

        # 获取高斯计算结果
        log_data = parse_gaussian_output(f"{logfiles_dir}/{mol_file.replace('.sdf', '.log')}")

        dataset.append({
            'smiles': smiles,
            'features': [
                log_data['homo'],
                log_data['lumo'],
                log_data['energy']
            ]
        })

    return pd.DataFrame(dataset)
