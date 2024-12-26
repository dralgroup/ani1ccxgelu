import sys
sys.path.append('/mlatom/home/fatemealavi/mlatom_devBranch')
import mlatom as ml
import mkl # required for KREG_API models
import matplotlib.pyplot as plt
import numpy as np  
import subprocess
import json as json
import csv
import tabulate
import pandas as pd
import torch
import torchani
import os
import math 
from tabulate import tabulate

print("ml:\n", ml)
molecule_formula = 'HCOOH' 
level_of_theory = "CCSD(T)-F12"

filename = "/mlatom/home/fatemealavi/anharmonic/data/meuwly_db_HCOOH_CCSDT_F12.npz"
data = np.load(filename)
labeled_database = ml.molecular_database.from_numpy(data['R'], data['Z'])
labeled_database.add_scalar_properties(data["E"] * 0.0367492929, property_name = "energy")
labeled_database.add_xyz_derivative_properties(-1 * data["F"] * 0.0367492929, property_name = "energy", xyz_derivative_property = "energy_gradients")

for mol in labeled_database.molecules:
    iflag = 0
    for iatom in range(len(mol.atoms)):
        if mol.atoms[iatom].element_symbol == "X":
            del(mol.atoms[iatom-iflag])
            iflag += 1

print("************************ ANI-1ccx_GELU **************************")
ani_1ccx_gelu = ml.models.model_tree_node(
    name='ani_gelu', 
    children=[ml.models.model_tree_node(
        name=f'nn{ii}',
        model=ml.models.ani(model_file=f'/mlatom/home/fatemealavi/anharmonic/ANI_1ccx_HCOOH_new_model/HCOOH_ANI_1ccx_TLOnCCSDT_F12_SP_energyShifter_FixL5and7/ani1ccx_gelu_TL_CCSDT_F12_HCOOH_byFateme/cv{ii}.pt'),
        operator='predict') for ii in range(8)],
    operator='average'
)

geoOptimization_anni1ccxGelu = ml.simulations.optimize_geometry(model = ani_1ccx_gelu, initial_molecule = labeled_database[0], program = 'gaussian')
optmol_ani1ccxGelu = geoOptimization_anni1ccxGelu.optimized_molecule
print("Optimized Geometry for Ani_1ccx_gelu:\n", optmol_ani1ccxGelu.get_xyz_coordinates())

# optmol_ani1ccxGelu.dump('optmol')
# print('\nStandard deviation of NNs (kcal/mol):')
# optmol_ani1ccxGelu.ani_gelu.standard_deviation(properties=['energy'])
# mol_std = optmol_ani1ccxGelu.ani_gelu.energy_standard_deviation * ml.constants.Hartree2kcalpermol
# print(mol_std)
# print('\n')
# exit()

ml.simulations.freq(model = ani_1ccx_gelu, program = 'gaussian', molecule = optmol_ani1ccxGelu, anharmonic = False)
frequencies_ani1ccxGelu = optmol_ani1ccxGelu.frequencies
numOfAtoms = len(labeled_database.molecules[0])
modes = [i + 1 for i in range(numOfAtoms * 3 - 6)]  # Nonlinear molecules
df_freq = pd.DataFrame(modes, columns = ["mode"])
df_freq["freq_ANI_1ccx_gelu"] = frequencies_ani1ccxGelu

print("****************** Frequencies of All Models **********************")
def update_database(df_freq):
    methods = ["exp", "mp2", "physnet_CCSD(T)-F12_NN1", "physnet_CCSD(T)-F12_NN2", "physnet_MP2_NN1", "physnet_MP2_NN2", "physnet_CCSD(T)_NN1", "physnet_CCSD(T)-NN2"]  
    df = pd.read_csv("/mlatom/home/fatemealavi/anharmonic/ANI_1ccx_HCOOH_new_model/HCOOH_ANI_1ccx_TLOnCCSDT_F12_SP_energyShifter_FixL5and7/freq_HCOOH_by_TL_ani_1ccx/freq_otherMethods.csv", sep = ",", header = 0)
    for meth in methods:    
        df_freq = pd.concat([df_freq, df[meth]], axis = 1)
    df_freq.set_index("mode", inplace = True)
    colHeaders = list(df_freq.columns.values)
    
    freqRMSEexpList = []
    freqMAEexpList = []
    freqRMSEmp2List = []
    freqMAEmp2List = []
    for col in colHeaders:
        freqRMSEexp = 0
        freqMAEexp = 0
        freqRMSEmp2 = 0
        freqMAEmp2 = 0
        for ind in modes:
            freqRMSEexp += (df_freq["exp"][ind] - df_freq[col][ind])**2
            freqMAEexp += abs(df_freq["exp"][ind] - df_freq[col][ind]) 
            freqRMSEmp2 += (df_freq["mp2"][ind] - df_freq[col][ind])**2
            freqMAEmp2 += abs(df_freq["mp2"][ind] - df_freq[col][ind])
        freqRMSEexp /= len(df_freq.index)
        freqRMSEexp = math.sqrt(freqRMSEexp)
        freqMAEexp /= len(df_freq.index)
        freqRMSEmp2 /= len(df_freq.index)
        freqRMSEmp2 = math.sqrt(freqRMSEmp2)
        freqMAEmp2 /= len(df_freq.index)
        freqRMSEexpList.append(freqRMSEexp)
        freqMAEexpList.append(freqMAEexp)     
        freqRMSEmp2List.append(freqRMSEmp2)
        freqMAEmp2List.append(freqMAEmp2)   
    df_freq.loc["MAE(exp)"] = freqMAEexpList 
    df_freq.loc["RMSE(exp)"] = freqRMSEexpList  
    df_freq.loc["MAE(mp2)"] = freqMAEmp2List 
    df_freq.loc["RMSE(mp2)"] = freqRMSEmp2List  
    
    return df_freq

freqOfDiffMethods = update_database(df_freq)

def generate_table_freq(freqOfDiffMethods):
    return tabulate(freqOfDiffMethods, headers = "keys", tablefmt = "psq1")
tableOfFreq = generate_table_freq(freqOfDiffMethods)
print("Frequencies of different models:\n", tableOfFreq)
with open ("/mlatom/home/fatemealavi/anharmonic/ANI_1ccx_HCOOH_new_model/HCOOH_ANI_1ccx_TLOnCCSDT_F12_SP_energyShifter_FixL5and7/freq_HCOOH_by_TL_ani_1ccx/harmonicfreq.txt", "w") as file1:
    file1.write(tableOfFreq)
    