State-aware Video Procedural Captioning
=====
PyTorch code and dataset for our ACM MM 2021 paper ["State-aware Video Procedural Captioning"](https://dl.acm.org/doi/abs/10.1145/3474085.3475322)
by [Taichi Nishimura](https://misogil0116.github.io/nishimura/), Atsushi Hashimoto, Yoshitaka Ushiku, Hirotaka Kameko, and Shinsuke Mori.

Video procedural captioning (VPC), which generates procedural text from instructional videos, is an essential task for scene understanding and real-world applications. The main challenge of VPC is to describe how to manipulate materials accurately. This paper focuses on this challenge by designing a new VPC task, generating a procedural text from the clip sequence of an instructional video and material list. In this task, the state of materials is sequentially changed by manipulations, yielding their state-aware visual representations (e.g., eggs are transformed into cracked, stirred, then fried forms). The essential difficulty is to convert such visual representations into textual representations; that is, a model should track the material states after manipulations to better associate the cross-modal relations. To achieve this, we propose a novel VPC method, which modifies an existing textual simulator for tracking material states as a visual simulator and incorporates it into a video captioning model. Our experimental results show the effectiveness of the proposed method, which outperforms state-of-the-art video captioning models. We further analyze the learned embedding of materials to demonstrate that the simulators capture their state transition.

## Getting started
### Prerequisites
0. Clone this repository
```
git clone https://github.com/misogil0116/svpc
cd svpc
```

1. Prepare feature files

Download [features.tar.gz](https://drive.google.com/file/d/1T5COAiqhIgqKvHzzsY2bw29fSuX68E39/view?usp=sharing) from Google drive.
The features/ directory stores ResNet + BN-Inception features for each video.
```
features
├── testing
├── training
├── validation
└── yc2
```

### Training and Inference
We give examples on how to perform training and inference.

1. Training
The general training command is:
```
bash scripts/train.sh MODEL_TYPE TEMP_PARAM, LAMBDA_PARAM, CHECKPOINT_DIR, FEATURE_DIR, DURATION_PATH
```
`MODEL_TYPE` can be one of `[vivt, viv, vi, v]`, see details below.
`TEMP_PARAM` and `LAMBDA_PARAM` is a gumbel softmax temperature parameter and lambda parameter, respectively (`TEMP_PARAM=0.5` and `LAMBDA_PARAM=0.5` work well in our experiments).
`CHECKPOINT_DIR`, `FEATURE_DIR`, and `DURATION_DIR` is checkpoint directory, feature directory, and duration csv filepath, respectively.

| MODEL_TYPE         | Description                            |
|--------------------|----------------------------------------|
| vivt               | +Visual simulator+Textual re-simulator |
| viv                | +Visual simulator                      |
| vi                 | Video+Ingredient                       |
| v                  | Video                                  |


To train VIVT model:
```
scripts/train.sh vivt 0.5 0.5 /path/to/model/checkpoint/ /path/to/features/ /path/to/duration_frame.csv
```

2. Evaluate trained model on word-overlap evaluation (BLEU, METEOR, CIDEr-D, and ROUGE-L)
```
scripts/eval_caption.sh MODEL_TYPE CHECKPOINT_PATH FEATURE_DIR DURATION_PATH
```
Note that you should specify checkpoint file (`.chkpt`) for `CHECKPOINT_PATH`.
Generated captions are saved at `/path/to/model/checkpoint/MODEL_TYPE_test_greedy_pred_test.json`.
This file is used for ingredient prediction evaluation.

3. Evaluate ingredient prediction
```
scripts/eval_ingredient_f1.sh MODEL_TYPE CAPTION_PATH
```
The results should be comparable with the results shown at Table 4 of the paper. 

4. Dump the learned embedding of ingredients
```
scripts/dump_embeddings.sh MODEL_TYPE CHECKPOINT_PATH FEATURE_DIR DURATION_PATH
```
This script generates `./MODEL_TYPE_step_embedding_dict.pkl`, which consists of material embedding at each step.

## Pretrained weights
You can download them from [here](https://drive.google.com/drive/folders/1S25vBqh_9MDf5XCw8svqhs5xaCv06Ccn?usp=sharing)

## Questions
- How to evaluate retrieval evaluation?

You can evaluate this by converting generated caption file (`CHECKPOINT_PATH`) into csv format that [MIL-NCE](https://github.com/antoine77340/MIL-NCE_HowTo100M) requests. See [here](https://github.com/antoine77340/MIL-NCE_HowTo100M#zero-shot-evaluation-retrieval-on-msr-vtt-and-youcook2) for additional information.

- How to access annotated ingredients?

You can access them [here](https://github.com/misogil0116/svpc/tree/master/densevid_eval/yc2_data).
The annotated ingredients are stored to the json files (see `ingredients` keys).

## Citation
If you use this code for your research, please cite our paper:
```
@inproceedings{taichi2021acmmm,
  title={State-aware Video Procedural Captioning},
  author={Taichi Nishimura and Atsushi Hashimoto and Yoshitaka Ushiku and Hirotaka Kameko and Shinsuke Mori},
  booktitle={ACMMM},
  pages={1766--1774},
  year={2021}
}
```

## Code base
This code is based on [MART](https://github.com/jayleicn/recurrent-transformer)

## Contact
taichitary [at] gmail.com.
