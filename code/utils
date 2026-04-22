import csv
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def get_l2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 1 * (param ** 2).sum()
    return reg


def get_dpiinformation(fpath):
    data = []
    with open(fpath, newline='') as csvfile:
        # 使用csv模块读取CSV文件
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        # 遍历CSV文件中的每一行数据
        for row in reader:
            # 处理每一行数据
            # print(', '.join(row))
            data.append(row)
    data = np.array(data)

    drug = []
    drug_smi = []
    for i, j in enumerate(data[:, 3]):
        if j not in drug:
            drug.append(j)
            drug_smi.append(data[i, 6])

    protein = []
    protein_seq = []
    for ii, jj in enumerate(data[:, 4]):
        if jj not in protein:
            protein.append(jj)
            protein_seq.append(data[ii, 7])
    return data, drug, drug_smi, protein, protein_seq


def get_data(fpath):
    data = []
    with open(fpath, newline='') as csvfile:
        # 使用csv模块读取CSV文件
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        # 遍历CSV文件中的每一行数据
        for row in reader:
            # 处理每一行数据
            # print(', '.join(row))
            data.append(row)
    data = np.array(data)

    drug_smi = []
    for i, j in enumerate(data[:, 0]):
        if j not in drug_smi:
            drug_smi.append(j)

    protein_seq = []
    for ii, jj in enumerate(data[:, 1]):
        if jj not in protein_seq:
            protein_seq.append(jj)
    return data, drug_smi, protein_seq


CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, ":":65, "~":65}

CHARISOSMILEN = 66


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]

    return X  # .tolist()


CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}
CHARPROTLEN = 25


def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]

    return X  # .tolist()



def morgan_smiles(line, dim_num):
    mol = Chem.MolFromSmiles(line)
    feat = AllChem.GetMorganFingerprintAsBitVect(mol, 2, dim_num)

    return feat


def nor_pro (S, t):
    P = np.zeros(S.shape)
    for i in range(len(S)):
        P[i]=S[i]/S[i,i]
        for j in range(len(S[0])):
            if P[i,j]<0:
                P[i, j]=0
    PP = ((P + P.T) / 2)**(1/t)
    return PP

# def sim_recon(S, t):
#     sorted_drug = (S).argsort(axis=1).argsort(axis=1)
#     np.fill_diagonal(sorted_drug, 0)
#     sorted_drug = (len(S) - 1) * np.ones((len(S), len(S))) - sorted_drug
#     sorted_drug[sorted_drug == 0] = 1
#     sorted_drug = 1 / ((sorted_drug) ** (1 / t))  # *(sorted_drug+1))
#     np.fill_diagonal(sorted_drug, 1)
#     SS = (sorted_drug + sorted_drug.T) / 2
#     return SS



def sim_recon(S, t):
    sorted_drug = (S).argsort(axis=1).argsort(axis=1)
    # np.fill_diagonal(sorted_drug, 0)
    sorted_drug = (len(S)) * np.ones((len(S), len(S))) - sorted_drug
    # sorted_drug[sorted_drug == 0] = 1
    sorted_drug = 1 / ((sorted_drug) ** (1 / t))  # *(sorted_drug+1))
    np.fill_diagonal(sorted_drug, 1)
    SS = (sorted_drug + sorted_drug.T) / 2
    return  SS



def get_pro_trans_list(protein, protein1, protein2):
    protein_list = list(protein1)
    num = protein_list.index(protein)
    submatrix = protein2[num]
    return submatrix
