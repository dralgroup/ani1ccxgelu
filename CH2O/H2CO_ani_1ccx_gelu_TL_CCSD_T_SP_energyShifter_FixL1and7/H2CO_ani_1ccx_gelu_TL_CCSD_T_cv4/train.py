import sys
sys.path.append('/export/home/fatemealavi/mlatom_devBranch')
import mlatom as ml
import mkl # required for KREG_API models
import numpy as np  
import subprocess
import json as json
import csv
import pandas as pd
import torch
import torchani
import os 
from mlatom.interfaces.torchani_interface import molDB2ANIdata
from mlatom.data import atomic_number2element_symbol

print("ml:\n", ml)

molecule_formula = 'H2CO' 
level_of_theory = "CCSD(T)"

filename = f"/mlatom/home/fatemealavi/anharmonic/data/ch2o_cc_avtz_3601.npz"
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

print("*************** Splitting data *********************")
labeled_database_train, labeled_database_test = ml.data.sample(molecular_database_to_split = labeled_database, sampling = 'random', number_of_splits = 2, split_equally=False, fraction_of_points_in_splits=[0.95, 0.05], indices=None)
data_test = labeled_database_test.copy()

print("************************ ANI-1ccx_GELU_single precision **************************")
ani_gelu = ml.models.ani(model_file='/mlatom/home/fatemealavi/anharmonic/ANI_1ccx_H2CO_new_model/H2CO_ANI_1ccx_TLOnCCSDT_SP_energyShifter_FixL1and7/H2CO_ANI_1ccx_TLonCCSDT_cv4/cv4.pt')
energy_shifter_HCNO = ani_gelu.energy_shifter # old energy shifter

energy_shifter_empty = torchani.utils.EnergyShifter(None) # create new energy shifter
molecule_species = [atomic_number2element_symbol[sp] for sp in np.unique(data['Z'])]
if 'X' in molecule_species:
    molecule_species.remove('X')
species_order_HCNO = ani_gelu.species_order

molDB2ANIdata(labeled_database_train, property_to_learn='energy', xyz_derivative_property_to_learn=None).subtract_self_energies(energy_shifter_empty, molecule_species) # extract self atomic energy for the training set

energy_shifter_new = []
for sp in species_order_HCNO:
    if sp not in molecule_species:
        energy_shifter_new.append(energy_shifter_HCNO.self_energies[species_order_HCNO.index(sp)])
    else:
        energy_shifter_new.append(energy_shifter_empty.self_energies[molecule_species.index(sp)])
        
ani_gelu.fix_layers([[0,6],[0,6],[0,6],[0,6]])
ani_gelu.energy_shifter = torchani.utils.EnergyShifter(energy_shifter_new) 
ani_gelu.train(molecular_database = labeled_database_train, property_to_learn = 'energy',  spliting_ratio = 0.895, xyz_derivative_property_to_learn = 'energy_gradients')

print("************************** Prediction *********************************")
ani_gelu.predict(molecular_database = labeled_database_test, calculate_energy = True, calculate_energy_gradients = True)

en_pre_ani1ccx_gelu_sp = np.array([mol.energy for mol in labeled_database_test.molecules])
np.savetxt(f"/mlatom/home/fatemealavi/anharmonic/ANI_1ccx_H2CO_new_model/H2CO_ANI_1ccx_TLOnCCSDT_SP_energyShifter_FixL1and7/H2CO_ANI_1ccx_TLonCCSDT_cv4/preEnOfAni1ccxGelu_sp.txt", en_pre_ani1ccx_gelu_sp)

en_ref_ani1ccx_gelu_sp = np.array([mol.energy for mol in data_test.molecules])
np.savetxt(f"/mlatom/home/fatemealavi/anharmonic/ANI_1ccx_H2CO_new_model/H2CO_ANI_1ccx_TLOnCCSDT_SP_energyShifter_FixL1and7/H2CO_ANI_1ccx_TLonCCSDT_cv4/refEnOfAni1ccxGelu_sp.txt", en_ref_ani1ccx_gelu_sp) 

force_pre_ani1ccx_gelu_sp = np.array([mol.get_energy_gradients() for mol in labeled_database_test.molecules])
np.save(f"/mlatom/home/fatemealavi/anharmonic/ANI_1ccx_H2CO_new_model/H2CO_ANI_1ccx_TLOnCCSDT_SP_energyShifter_FixL1and7/H2CO_ANI_1ccx_TLonCCSDT_cv4/preForceOfAni1ccxGelu_sp.npy", force_pre_ani1ccx_gelu_sp)

force_ref_ani1ccx_gelu_sp = np.array([mol.get_energy_gradients() for mol in data_test.molecules])
np.save(f"/mlatom/home/fatemealavi/anharmonic/ANI_1ccx_H2CO_new_model/H2CO_ANI_1ccx_TLOnCCSDT_SP_energyShifter_FixL1and7/H2CO_ANI_1ccx_TLonCCSDT_cv4/refForceOfAni1ccxGelu_sp.npy", force_ref_ani1ccx_gelu_sp)

ermse_ani1ccx_gelu_sp = np.sqrt(np.mean((en_pre_ani1ccx_gelu_sp - en_ref_ani1ccx_gelu_sp)**2))
emae_ani1ccx_gelu_sp = np.mean(np.abs(en_pre_ani1ccx_gelu_sp - en_ref_ani1ccx_gelu_sp))
frmse_ani1ccx_gelu_sp = np.sqrt(np.mean((force_pre_ani1ccx_gelu_sp - force_ref_ani1ccx_gelu_sp)**2))
fmae_ani1ccx_gelu_sp = np.mean(np.abs(force_pre_ani1ccx_gelu_sp - force_ref_ani1ccx_gelu_sp))

accuracy = pd.DataFrame(["EMAE", "ERMSE", "FMAE", "FRMSE"], columns = ["accuracy"])
accuracy["ANI_1ccx_gelu_sp"] = [ermse_ani1ccx_gelu_sp, emae_ani1ccx_gelu_sp, frmse_ani1ccx_gelu_sp, fmae_ani1ccx_gelu_sp]
print("accuracy:\n", accuracy)
