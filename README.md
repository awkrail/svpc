State-aware Video Procedural Captioning
=====
PyTorch code and dataset for our ACM MM 2021 paper ["State-aware Video Procedural Captioning"]()
by [Taichi Nishimura](https://misogil0116.github.io/nishimura/)

This paper proposes a novel approach to generate a procedural text from the clip sequence pre-segmented in an instructional video and material list.
The essential difficulty is to convert such visual representations into textual representations; that is, a model should track the material states after manipulations to better associate the cross-modal relations.
To achieve this, we propose a novel VPC method, which modifies an existing textual simulator for tracking material states as a visual simulator and incorporates it into a video captioning model.
Our experimental results show the effectiveness of the proposed method, which outperforms state-of-the-art video captioning models.
We further analyze the learned embedding of materials to demonstrate that the simulators capture their state transition.

## Getting started
### Prerequisites
0. Clone this repository
```
git clone https://github.com/misogil0116/svpc
cd svpc
```

1. Prepare feature files

Download features from the [YouCook2 page](http://youcook2.eecs.umich.edu/download)
```
wget http://youcook2.eecs.umich.edu/static/YouCookII/YouCookII.tar.gz
tar -xvzf path/to/rt_yc2_feat.tar.gz 
```

### Training and Inference
We give examples on how to perform training and inference.

0. Build Vocabulary
```
bash scripts/build_vocab.sh /path/to/glove.6B.300d.txt
```

1. Training

The general training command is:
```
bash scripts/train.sh DATASET_NAME MODEL_TYPE
```
`MODEL_TYPE` can be one of `[vivt, viv, vi, v]`, see details below.

| MODEL_TYPE         | Description                            |
|--------------------|----------------------------------------|
| vivt               | +Visual simulator+Textual re-simulator |
| viv                | +Visual simulator                      |
| vi                 | Video+Ingredient                       |
| v                  | Video                                  |


To train full model:
```
bash scripts/train.sh vivt 0.5 0.5 /path/to/model/checkpoint/ /path/to/features/ /path/to/duration_frame.csv
```

2. Generate captions 
```
bash scripts/translate_greedy.sh anet_re_* val
```
Replace `anet_re_*` with your own model directory name. 
The generated captions are saved at `results/anet_re_*/greedy_pred_val.json`


3. Evaluate generated captions
```
bash scripts/eval.sh anet val results/anet_re_*/greedy_pred_val.json
```
The results should be comparable with the results we present at Table 2 of the paper. 
E.g., B@4 10.33; R@4 5.18.

## Citation
If you use this code for your research, please cite our paper:
```
@inproceedings{taichi2021acmmm,
  title={State-aware Video Procedural Captioning},
  author={Taichi Nishimura and Atsushi Hashimoto and Yoshitaka Ushiku and Hirotaka Kameko and Shinsuke Mori},
  booktitle={ACMMM},
  year={2021}
}
```

## Code base
This code is based on [MART](https://github.com/jayleicn/recurrent-transformer)

## Contact
taichitary [at] gmail.com.
