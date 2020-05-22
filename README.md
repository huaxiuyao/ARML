# ARML (Automated Relational Meta-learning)

## About
Source code<a href="#note1" id="note1ref"><sup>1</sup></a> of the paper [Automated Relational Meta-learning](https://openreview.net/forum?id=rklp93EtwH)


If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{
  yao2020automated,
  title={Automated Relational Meta-learning},
  author={Huaxiu Yao and Xian Wu and Zhiqiang Tao and Yaliang Li and Bolin Ding and Ruirui Li and Zhenhui Li},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=rklp93EtwH}
}
```

## Data
Plain-Multi and Art-Multi are released in [Google Drive](https://drive.google.com/drive/folders/1I35LjOO8tRCb8fevpxEZJdYZiIZWRLz6?usp=sharing)

## Usage
### Dependence
* python 3.*
* TensorFlow 1.10+
* Numpy 1.15+

### 2-D Regression
Please see the bash file in /2D_bash for hyperparameter settings

### Plainmulti
Please see the bash file in /plainmulti_bash for hyperparameter settings

### Artmulti
Please see the bash file in /artmulti_bash for hyperparameter settings

<a id="note1" href="#note1ref"><sup>1</sup></a>This code is built based on the [MAML](https://github.com/cbfinn/maml) and [HSML](https://github.com/huaxiuyao/HSML).
