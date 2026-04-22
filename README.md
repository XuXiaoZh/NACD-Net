# Getting Start

1. Environment Setup

It is recommended to use conda to manage dependencies:
    conda create -n v3 python=3.8

    conda activate v3

    pip install -r requirements.txt


2. Data Preparation

Organize your dataset under transfer/data/ with the following structure:

    transfer/data/
    ├── train/
    ├── val/
    └── test/
    
Data source:

https://github.com/smousavi05/STEAD

Run the debug script to verify data loading:
    
    python transfer/debug_data.py
3. Training & Fine-Tuning

# Example: Run training for the base v3 model

    python exp2_freeze_v3/train.py

# Fine-tune the model with transfer learning
    python transfer/retrain.py
4. Validation & Visualization

        python transfer/val.py

# Single transfer model validation
    python transfer/val_trans.py

# Batch validation for all transfer models

    python transfer/val_trans_all.py

Visualization of Results

    python transfer/keshihua.py

Validation plots will be saved automatically to transfer/val_plots/.

📊 Experiment Overview

exp2_freeze_* series:These experiments evaluate the impact of different layer-freezing strategies on model performance. fixed, v2, and v3 represent successive improvements and corrections to the original setup.

exp3_domain_shift/:Contains experiments on domain adaptation and generalization across different data distributions.

transfer/ series:The core pipeline for transfer learning, including fine-tuning, validation, and ablation studies on different datasets, data sizes, and preprocessing methods (e.g., wavelet transforms).

📝 Notes

Model checkpoints are automatically saved to the corresponding checkpoints_* directories.

All validation and visualization outputs are stored in transfer/val_plots/.

New experiments can be built upon existing scripts such as expr3.py or xiaorong_v2.py by modifying parameters and configurations.

📧 Contact & Maintenance

Email: [x_xiaozheng0908@163.com]

For questions, suggestions, or collaboration, please feel free to reach out.
