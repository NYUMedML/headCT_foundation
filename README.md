# HeadCT-Foundation (Nature Biomedical Enginnering 2026)
[![Paper](https://img.shields.io/badge/Paper-Nat.%20Biomed.%20Eng.-831C91)](https://www.nature.com/articles/s41551-026-01668-w)
[![Paper](https://img.shields.io/badge/Paper-arxiv.2502.02779-FF6B6B.svg)](https://arxiv.org/abs/2502.02779)
</div>

**Abstract:** Head computed tomography (CT) is a widely-used imaging modality for assessing brain, skull, and cerebrovascular pathologies, particularly in neurologic emergencies due to its speed, safety, and accessibility. However, its limited sensitivity compared to MRI and the scarcity of annotated data hinder the development of robust diagnostic models. To address this, we propose a novel head CT foundation model using self-supervised learning on 361,663 non-contrast 3D head CT scans. Our approach leverages self-supervised learning to pre-train a model that learns generalizable features from unlabeled data, followed by fine-tuning on smaller annotated datasets for tasks like hemorrhage and tumor detection. Evaluated on internal and external datasets, the model demonstrates superior performance on downstream tasks and strong generalization across in- and out-of-distribution data. This work establishes a new benchmark for head CT analysis, highlighting the potential of scaling self-supervised learning in 3D medical imaging.

<img src="./images/overview.png" width="900px"/>

## Installation
1. Create environment with conda:
```
conda create -n head_ct python=3.8.18 -y
conda activate head_ct
```
2. Clone the repository:
```
git clone https://github.com/NYUMedML/HeadCT-Foundation
cd headCT-Foundation
```
3. Install dependencies:
```
pip install --upgrade pip
pip install -r requirement.txt
pip install -e .
```

## Starter Notebook
See [**./notebooks/extract_feature_sample.ipynb**](./notebooks/extract_feature_sample.ipynb) to get started of how to load data, model weights and extract features from head CT scans. See [**./notebooks/visualization_sample.ipynb**](./notebooks/visualization_sample.ipynb) to find how to load and visualize volumetric head CT scan.

## Train Model
We present how to run different downstream training and pre-training methods in root directory of this repository with different specified hyperparameters. For more details on pre-defined hyperparameters and their usage, please check [**./config.py**](./config.py) and called ``yaml`` files (e.g.  [**./configs/downstream/vit_HeadCT_cq500.yaml**](./configs/downstream/vit_HeadCT_cq500.yaml) for ``--cfg ./configs/downstream/vit_HeadCT_cq500.yaml``)

We additionally present examples of submitting jobs to Slurm Workload Manager in [**./slurm_submit**](./slurm_submit)

### Train Model for Downstream

#### Fine-tuning
```
torchrun --nnodes 1 --nproc_per_node 1 --master_port 12400 ./main_downstream.py --local_rank 0 \
    --model_name "vit" --batch_size 64 --num_workers 4 --max_epochs 10 --base_lr 1e-5 \
    --cfg ./configs/downstream/vit_HeadCT_cq500.yaml \
    --optimizer "AdamW" --scheduler "cosine" --weight_decay 0.01 --grad_clip 1.0 \
    --preds_save_name "cq500_ICH_finetune" --use_amp \
    --classifier "linear" --label_name "ICH" --dataset "cq500" --seed 42 \
    --filename "cq500_ICH"
```

#### Linear-probing
Add ``--freeze`` to command arguments will only update weights for classification layer:
```
torchrun --nnodes 1 --nproc_per_node 1 --master_port 12400 ./main_downstream.py --local_rank 0 \
    --model_name "vit" --batch_size 64 --num_workers 4 --max_epochs 10 --base_lr 1e-5 \
    --cfg ./configs/downstream/vit_HeadCT_cq500.yaml \
    --optimizer "AdamW" --scheduler "cosine" --weight_decay 0.01 --grad_clip 1.0 \
    --preds_save_name "cq500_ICH_linear_prob" --use_amp \
    --classifier "linear" --label_name "ICH" --dataset "cq500" --seed 42 \
    --filename "cq500_ICH" --freeze
```

#### Few-shots
Add ``--few_shots <num_shots>`` to command arguments will perform few shots with ``<num_shots>`` positive samples for selected disease:
```
torchrun --nnodes 1 --nproc_per_node 1 --master_port 12400 ./main_downstream.py --local_rank 0 \
    --model_name "vit" --batch_size 64 --num_workers 4 --max_epochs 10 --base_lr 1e-5 \
    --cfg ./configs/downstream/vit_HeadCT_cq500.yaml \
    --optimizer "AdamW" --scheduler "cosine" --weight_decay 0.01 --grad_clip 1.0 \
    --preds_save_name "cq500_ICH_fewshots_8" --use_amp \
    --classifier "linear" --label_name "ICH" --dataset "cq500" --seed 42 \
    --filename "cq500_ICH" --few_shots 8
```

### Pre-train Model

#### DINO Pre-training
```
torchrun --nnodes 1 --nproc_per_node 1 --master_port 12400 ../main_pretrain_dino.py --local_rank 0 \
    --model_name "dino" --batch_size 64 --num_workers 4 --max_epochs 1000 --base_lr 1.5e-4 \
    --cfg ../configs/dino/dino_HeadCT.yaml \
    --use_amp --optimizer "AdamW" --scheduler "cosine" --weight_decay 5e-3 \
    --grad_clip 3.0
```

#### MAE Pre-training
```
torchrun --nnodes 1 --nproc_per_node 1 --master_port 12400 ../main_pretrain_mae.py --local_rank 0 \
    --model_name "mae" --batch_size 256 --num_workers 4 --max_epochs 400 --base_lr 1.5e-4 \
    --cfg ../configs/mae/mae_HeadCT.yaml \
    --use_amp --optimizer "AdamW" --scheduler "cosine" --weight_decay 5e-3 \
    --grad_clip 3.0
```

## Comparisons and Benchmarks
We present comparison among our model, [**Merlin**](https://arxiv.org/abs/2406.06512) (a 3D vision-language foundation model pre-trained on large-scale abdominal CT), [**Google CT Foundation**](https://github.com/Google-Health/imaging-research/tree/master/ct-foundation) (a 3D CT vision foundation model pre-trained on large-scale CT of different anatomy), [**CT-FM**](https://github.com/project-lighter/CT-FM) (another 3D CT vision foundation model pre-trained with different CT anatomy) and model trained from scratch in the plots on multi-label diseases diagnosis and hemorrhage subtypes volume-to-volume retrieval tasks, where we show our model outperforms others across the board. This highlights the success of our training pipeline and importance of our domain specific large-scale pre-training. We additional show comparison on alternative modeling methods on Attention-Based Multiple-Instance-Learning (AB-MIL) and Mean Pooling with 2D foundation model (DINOv3) and directly using 3D Video Foundation Model (VJEPA2), where we show consistent improved performance on our model.

#### Fine-Tuning Comparison <br>
<img src="./images/performance1.png" width="900px"/>

#### Linear Probing Comparison ([**Google CT Foundation**](https://github.com/Google-Health/imaging-research/tree/master/ct-foundation) only allow API access) <br>
<img src="./images/performance2.png" width="475px" alt="Performance Image" style="display: block; margin-left: 160px;">

#### Volume-to-Volume Retrieval Comparison <br>
<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="./images/mAP_RSNA_Retrieval.png" width="240px" alt="RSNA Retrieval" />
  <img src="./images/mAP_CQ500_Retrieval.png" width="310px" alt="CQ500 Retrieval" />
</div>

#### Alterntaive Modeling Methods Comparison <br>
<img src="./images/performance3.png" width="650px" alt="Alterntiave 1" style="display:inline-block; margin: 0 10px;" />
<img src="./images/performance4.png" width="650px" alt="Alterntiave 2" style="display:inline-block; margin: 0 10px;" />

## Attention Map Visualization
We present our model attention map visualization here across slices of scan for different diseases, where our model can attend to important region of diagnosing diseases.

<img src="./images/attention_map.png" width="800px"/>

## Scans Filtering Criterion
We present the filtering criterion in combination of Study Description, Kilovoltage Peak (kVp) and Convolution Types for selecting relevant high quality CT scans on building our foundation model in [**./scans_filter_criterion/scans_filter_criterion.csv**](./scans_filter_criterion/scans_filter_criterion.csv).

## Datasets
Dataset splits with labels for CQ500 and RSNA are organized in their respective directories under [**./datasets**](./datasets), with the root directory removed from the image paths to facilitate reproducing our experimental results.

## Model Weights Sharing
Due to the possibility of inferring private patient facial features from Head CT data, we apologize that public release of the model weights is not permitted. The model weights are only available upon request after signing institutional agreement. Requests for model weights should be sent to the corresponding author and the NYU Langone Data Sharing Strategy Board (DSSB) Committee (DataSharing@nyulangone.org).

## Citation
If you find this repository useful and want to see more details, please consider citing and checking our preprint paper:
```
@Article{Zhu2026,
    author={Zhu, Weicheng and Huang, Haoxu and Tang, Huanze and Musthyala, Rushabh and Yu, Boyang and Chen, Long and Vega, Emilio and O'Donnell, Thomas and Hayek, Reya and Kuohn, Lindsey and Dehkharghani, Seena and Frontera, Jennifer A. and Masurkar, Arjun V. and Melmed, Kara and Razavian, Narges},
    title={3D Foundation Model for Generalizable Disease Detection in Head Computed Tomography},
    journal={Nature Biomedical Engineering},
    year={2026},
    month={Apr},
    day={22},
    issn={2157-846X},
    doi={10.1038/s41551-026-01668-w},
    url={https://doi.org/10.1038/s41551-026-01668-w}
}
or
@Article{zhu2025foundationctmodel, 
    title={3D Foundation Model for Generalizable Disease Detection in Head Computed Tomography}, 
    author={Zhu, Weicheng and Huang, Haoxu and Tang, Huanze and Musthyala, Rushabh and Yu, Boyang and Chen, Long and Vega, Emilio and O'Donnell, Thomas and Hayek, Reya and Kuohn, Lindsey and Dehkharghani, Seena and Frontera, Jennifer A. and Masurkar, Arjun V. and Melmed, Kara and Razavian, Narges}, 
    year={2025}, 
    eprint={2502.02779}, 
    archivePrefix={arXiv}, 
    primaryClass={eess.IV}, 
    url={https://arxiv.org/abs/2502.02779},
    note={Weicheng Zhu and Haoxu Huang contributed equally to this work}
}
```
<img src="./images/logo.png" width="700px"/>