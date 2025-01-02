This repository contains models and scripts for:

- Seyedeh Fatemeh Alavi, Yuxinxin Chen, Yi-Fan Hou, Fuchun Ge, Peikun Zheng, [Pavlo O. Dral](http://dr-dral.com)*. [ANI-1ccx-gelu Universal Interatomic Potential and Its Fine-Tuning: Toward Accurate and Efficient Anharmonic Vibrational Frequencies](https://doi.org/10.1021/acs.jpclett.4c03031). *J. Phys. Chem. Lett.* **2025**, in press. DOI: 10.1021/acs.jpclett.4c03031. Preprint on ChemRxiv: https://doi.org/10.26434/chemrxiv-2024-c8s16 (2024-10-09).

Please also check the related [tutorials](https://xacs.xmu.edu.cn/docs/mlatom/tutorial_universalml.html#fine-tuning-universal-models) on using [MLatom](https://github.com/dralgroup/mlatom) for calculations with the universal models and their fine-tuning.

# Models and scripts

This repository contains codes and scripts for utilizing and fine-tuning a reformulated Universal Interatomic Potential (UIP), **ANI-1ccx-gelu**, to calculate anharmonic vibrational frequencies of various molecules, including H2CO, HCOOH, HNO2, cH3OH, CH3CHO, CH3NO2, CH3COOH, and CH3CONH3. While ANI-1ccx demonstrates excellent thermochemical performance, its limitations in anharmonic frequency calculations are addressed in this enhanced version. This UIP represents a significant step forward in using machine learning for interpreting molecular spectra, reducing computational cost while maintaining accuracy comparable to B3LYP/6-31G*.

## Key Features

- **ANI-1ccx-gelu model** consists of an ensemble of eight models, similar to ANI-1ccx, which improves the model's accuracy. You can find this model in a folder with the same name.

- The main programs are located in `Scripts/master.py` and `Scripts/train.py`:
    - **train.py**: The main script for retraining the ANI-1ccx-gelu model.
    - **master.py**: The main script for calculating anharmonic (or harmonic) frequencies using the TL ANI-1ccx-gelu model.

## Files & Inputs

- **"filename"**: Contains data for molecules located in the **Data Folder**. The data is taken from a previous study (see https://pubs.acs.org/doi/10.1021/acs.jctc.1c00249)
- **"model_file"**: Refers to the ANI-1ccx-gelu model, which is stored in the **ani-1ccx-gelu Folder**.  
- **freq_otherMethods.csv**: Includes calculated anharmonic frequencies using MP2 ab initio calculations, PhysNet models (NN1, NN2), and experimental data.

## How to Run the Scripts

To run scripts, use Python that meets the basic requirements available at [mlatom.com](http://mlatom.com/download/). 

- To fine-tune the ANI-1ccx-gelu model for a new problem (specific molecule and/or level of theory), run the `train.py` script. For example, you can view the calculations for all molecules in this project in folders named after the respective molecules. For instance, for CH2O molecule, there are three subfolders corresponding to three datasets: MP2, CCSD(T), and CCSD(T)-F12.

- For each of these calculations, fine-tuning is necessary for each of the eight models in ANI-1ccx-gelu by running the `train.py` script, as shown for example in the folder "H2CO_ani_1ccx_gelu_TL_CCSD_T_F12_cv0". Before runing train.py, change `filename` and `model_file` to the paths of your desired data and model. The eight TL models created from fine-tuning the ani-1ccx-gelu are located in a folder named "ani1-ccx-gelu_TL_CCSD_T_F12".

- Using the fine-tuned ANI-1ccx-gelu model, you can calculate anharmonic frequencies with the `master.py` script. 

- **Note**: To calculate harmonic frequencies, simply set the `anharmonic` parameter to `False` in the `ml.simulation` module.

## Outputs

- The **"anharmonicfreq.txt"** file contains the calculated anharmonic frequencies, as well as other methods for comparison.

- For detailed methodologies, refer to the above-mentioned paper.

## Platform Support

- **MLatom** | **XACScloud**
