# Probabilistic Compositional Embeddings for Multimodal Image Retrieval

Official repo for the implementation of [Probabilistic Compositional Embeddings for Multimodal Image Retrieval](https://arxiv.org/).

Presented at the [Multimodal Learning and Applications Workshop @ CVPR 2022](https://mula-workshop.github.io/).

## Introduction

Existing works in image retrieval often consider retrieving images with one or two query inputs, which do not generalize to multiple queries. In this work, we investigate a more challenging scenario for composing multiple multimodal queries in image retrieval. Given an arbitrary number of query images and (or) texts, our goal is to retrieve target images containing the semantic concepts specified in multiple multimodal queries. 

<img src="https://user-images.githubusercontent.com/23747497/162187533-f99ebee2-77a3-47b1-be08-464e9eb4db6d.png" width="60%">

To learn an informative embedding that can flexibly encode the semantics of various queries, we propose a novel multimodal probabilistic composer (MPC). Specifically, we model input images and texts as probabilistic embeddings, which can be further composed by a probabilistic composition rule to facilitate image retrieval with multiple multimodal queries. 

<img src="https://user-images.githubusercontent.com/23747497/162187988-e0346d9f-9183-49cd-a968-c582811a4a25.png" width="60%">

We propose a new benchmark based on the MS-COCO dataset and evaluate our model on various setups that compose multiple images and (or) text queries for multimodal image retrieval. Without bells and whistles, we show that our probabilistic model formulation significantly outperforms existing related methods on multimodal image retrieval while generalizing well to query with different amounts of inputs given in arbitrary visual and (or) textual modalities. 

## Setup

To setup the environment to run the code create a conda environment using the following commands:
```
conda env create -f environment.yml
conda activate mpc_env
```

### Data preparation
Download the files 'captions_train2017' and 'instances_train2017' from the [COCO website](https://cocodataset.org/#download) into the datasets/coco folder.

## Training and testing
The config files that control the execution can be found in the code/resources/configs/ folder. The file probabilistic.yaml is used to train the probabilistic models (MPC, PCME+addition) and the file deterministic.yaml is used to run the baselines from the paper (TIRG, MRN, FiLM, Relationship). To run the training, run the following command (example for probabilistic.yaml):
```
python code/main.py --config_name=probabilistic.yaml --mode=train
```

## COCO benchmark

The files and generating code for the COCO benchmark can be found in datasets/coco. Additional info is provided in the respective README file.

## Acknowledgements

- The probabilistic embedders for image and text are based on the implementation in [PCME](https://github.com/naver-ai/pcme).
- The TIRG image/text models and combiner as well as the evaluation code are based on the TIRG implementation in [TIRG code](https://github.com/google/tirg).

## Bibtex

    @inproceedings{neculai2022probabilistic,
      title={Probabilistic Compositional Embeddings for Multimodal Image Retrieval},
      author={Neculai, Andrei and Chen, Yanbei and Akata, Zeynep},
      booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition(CVPR) Workshops},
      year={2022},
    }
