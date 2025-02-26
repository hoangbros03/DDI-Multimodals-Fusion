# DDI-Multimodals-Fusion

This repository contains the code and configurations for multimodal fusion-based drug-drug interaction (DDI) prediction. The approach integrates various modalities such as text descriptions, molecular formulas, images, and graph structures.

## Installation

To set up the repository and install dependencies, run the following commands:

```bash
!git clone https://github.com/hoangbros03/DDI-Multimodals-Fusion.git
%cd DDI-Multimodals-Fusion
!pip install -r requirements.txt --quiet
!pip install -e . --quiet
!pip install rdkit torch_geometric --quiet

!git lfs install
!git clone https://huggingface.co/hoangtb203/DDI-Multimodals-Fusion-2024
```

## Configuration

Modify or create a new configuration file based on the sample `config_test.yaml`. The file includes paths for dataset files, model parameters, and training configurations.

### Sample Configuration (config_test.yaml)

```yaml
%%writefile config_test.yaml

# PLEASE CREATE NEW CONFIG INSTEAD OF OVERRIDE THIS CONFIG
# Change this string to change the folder name of saving models
training_session_name: session_name

# Lookup
type_embed: 'asada_bert_unpad'
model_name_or_path: './DDI-Multimodals-Fusion-2024'
candidate_train_path: "cache/pkl/v2/notprocessed.candidates.train.pkl"
candidate_test_path: "cache/pkl/v2/notprocessed.candidates.test.pkl"
desc_dict_path: "./DDI-Multimodals-Fusion-2024/desc_pretrained.pt"
image_data_train_path: "./DDI-Multimodals-Fusion-2024/data_image_train_allmodal.pt" 
image_data_test_path: "./DDI-Multimodals-Fusion-2024/data_image_test_allmodal.pt"
map_drug_to_formula_csv_path: 'cache/mapped_drugs/DDI/ease_matching/full.csv'
test_filtered_index_path : 'cache/filtered_ddi/test_filtered_index.txt'

# Saved customdataset
train_custom_dataset: 'ddi_train.pt'
test_custom_dataset: 'ddi_test.pt'

# Train configs
seed: 129
desc_conv_output_size: 64
desc_conv_window_size: 3
desc_layer_hidden: 0
formula_conv_output_size: 64 
formula_conv_window_size: 4
use_graph: True
use_formula: True
use_desc: True
use_image: True
loss_type: ddi_no_negative
freeze_bert: True
min_batch_size: 9
out_conv_list_dim: 256
fusion_module: concat
batch_size: 128
weight_decay: 0.01
lr: 0.0001
epochs: 50
represent_dim: [48,48,48,48]
conv_window_size: [3, 5, 7]
device: "cuda"
parallel_training: False
target_class: 5
activation: 'gelu'
pos_emb_dim: 10
max_seq_length: 256
optimizer: 'adamw'
dropout_rate_other: 0.5
dropout_rate_bert_output: 0.3
middle_layer_size: 128
gnn_option: 'GCN'
gnn_readout_option: 'global_max_pool'
gnn_num_layers: 5
gnn_hidden_channels: 32
gnn_use_gat_v2: False
gnn_use_edge_attr: True
apply_mask: False
freeze_gnn: True
freeze_formula: False
freeze_image: True
freeze_desc: False

gnn_state_path: './DDI-Multimodals-Fusion-2024/gcn_ck.pt'
formula_state_path: './DDI-Multimodals-Fusion-2024/formula_ck.pt'
desc_state_path: './DDI-Multimodals-Fusion-2024/desc_ck.pt'
image_state_path: ''
```

## Training

Once the environment is set up and the configuration file is in place, start training with the following command:

```bash
!python3 scripts/run_train.py --yaml_path config_test.yaml
```

## Features
- Multimodal fusion-based DDI prediction.
- Supports graph neural networks (GNNs) for structural data.
- Uses transformer-based text embeddings.
- Custom dataset handling and flexible configurations.

## Citation
If you use this repository for research purposes, please cite the corresponding paper (if available).

---

For any issues or inquiries, please contact the repository maintainers.

