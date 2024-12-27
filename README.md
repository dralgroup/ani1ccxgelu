# Models and scripts

This repository contains models and scripts for:

- Seyedeh Fatemeh Alavi, Yuxinxin Chen, Yi-Fan Hou, Fuchun Ge, Peikun Zheng, Pavlo O. Dral. *Towards Accurate and Efficient Anharmonic Vibrational Frequencies with the Universal Interatomic Potential ANI-1ccx-gelu and Its Fine-Tuning*. **2024**. Preprint on ChemRxiv: https://doi.org/10.26434/chemrxiv-2024-c8s16 (2024-10-09).

This repository contains codes and scripts for utilizing and fine-tuning a reformulated Universal Interatomic Potential (UIP), ANI-1ccx-gelu, for calculating anharmonic vibrational frequencies of some molecules, including H2CO, HCOOH, HNO2, cH3OH, CH3CHO, CH3NO2, CH3COOH, CH3CONH3. While ANI-1ccx demonstrates excellent thermochemical performance, its shortcomings in anharmonic frequency calculations are addressed in this enhanced version. This UIP is a significant step in advancing machine learning techniques for interpreting molecular spectra, reducing computational cost while maintaining reasonable accuracy comparable to B3LYP/6-31G*.

- ANI-1ccx-gelu model consists of an ensemble of eight models as ANI-1ccx, which improves the accuracy of the model. You can find this model in a folder with the same name.

- The main program is located in the Scripts/master.py and Scripts/train.py
train.py: main script to retrain ani-1ccx-gelu
master.py: mamin script to calculate anharmonic (or harmonic) frequencies using TL ani-1ccx-gelu model

- To run these scripts, please use Python which satisfies the basic requirements that can be found at http://mlatom.com/download/. For fine-tuning ani-1ccx-gelu model, please run /Script/train.py for a new problem: a specific molecule and/or level of theory. For instance, You can view the calculations for all molecules in this project in folders named after the respective molecules. For example, for CH2O, there are three subfolders corresponding to three datasets: MP2, CCSD(T), and CCSD(T)-F12. For each of these calculations, it is necessary to fine-tune each of the 8 models in ani-1ccx-gelu by train.py script as you can see for example in every folder named H2CO_ani_1ccx_gelu_TL_CCSD_T_F12_cv{}. The 8 fine-tuned models are located in a folder named "ani1-ccx-gelu_TL_CCSD_T_F12". Now using fine-tune ani-1ccx-gelu you can calculate anharmonic frequencis master.py script. Note to calculate harmonic frequencies, simply set the "anharmonic" parameter to "False" in the "ml.simulation" module.


- For detailed methodologies, refer to the considered paper.
- Platform Support: MLatom | XACScloud