from rdkit import Chem
from rdkit.Chem import Descriptors


def validate_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False

    # 验证基本化学规则
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False


def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol)
    }