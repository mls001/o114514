import os
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
from cclib.io import ccread
from cclib.parser.utils import convertor


def run_gaussian_opt_freq(smiles, method="B3LYP", basis_set="6-31G*", nproc=4, mem="4GB"):
    """
    输入SMILES字符串，生成3D结构，运行高斯优化和频率计算，返回结果。

    参数:
        smiles (str): 分子SMILES
        method (str): 计算方法，如B3LYP
        basis_set (str): 基组，如6-31G*
        nproc (int): 使用的CPU核心数
        mem (str): 内存分配

    返回:
        dict: 优化能量、频率、热力学数据等
    """
    # 1. 生成分子3D结构
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效的SMILES字符串")
    mol = Chem.AddHs(mol)  # 添加氢原子
    AllChem.EmbedMolecule(mol)  # 生成3D构型
    AllChem.MMFFOptimizeMolecule(mol)  # 预优化（MMFF力场）

    # 2. 生成高斯输入文件
    input_content = f"""%NProcShared={nproc}
%Mem={mem}
#P {method}/{basis_set} Opt Freq

Title

0 1
{AllChem.MolToXYZBlock(mol)}
"""
    with open("input.com", "w") as f:
        f.write(input_content)

    # 3. 运行高斯计算
    try:
        subprocess.run(["g16", "input.com"], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"高斯计算失败: {e}")

    # 4. 解析结果
    data = ccread("input.log")
    results = {
        "optimized_energy": convertor(data.scfenergies[-1], "eV", "kJ/mol"),  # 优化后的能量（kJ/mol）
        "frequencies": data.vibfreqs,  # 振动频率（cm⁻¹）
        "enthalpy": convertor(data.enthalpy, "eV", "kJ/mol"),  # 焓（kJ/mol）
        "free_energy": convertor(data.freeenergy, "eV", "kJ/mol"),  # 自由能（kJ/mol）
        "thermochemical_data": data.temperature  # 温度（默认298.15 K）
    }
    return results


# 示例使用
if __name__ == "__main__":
    smiles = "CCO"  # 乙醇
    try:
        results = run_gaussian_opt_freq(smiles, method="B3LYP", basis_set="6-31G*", nproc=4)
        print("优化能量 (kJ/mol):", results["optimized_energy"])
        print("振动频率 (cm⁻¹):", results["frequencies"][:5])  # 输出前5个频率
        print("焓 (kJ/mol):", results["enthalpy"])
        print("自由能 (kJ/mol):", results["free_energy"])
    except Exception as e:
        print(f"错误: {e}")
