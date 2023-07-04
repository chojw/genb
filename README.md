# Generative Bias for Robust Visual Question Answering

This repo contains the PyTorch code release for our paper GenB: [**Generative Bias for Robust Visual Question Answering**](https://arxiv.org/abs/2208.00690) (CVPR 2023).

### Prerequisites

Please make sure you are using a NVIDIA GPU with Python==3.7.1 and about 100 GB of disk space.

Install all requirements with ``pip install -r requirements.txt``


### Data Setup

Download UpDn features from [google drive](https://drive.google.com/drive/folders/111ipuYC0BeprYZhHXLzkRGeYAHcTT0WR?usp=sharing), which is the link from [this repo](https://github.com/GeraldHan/GGE), into ``/data/detection_features`` folder

Download questions/answers for VQA v2 and VQA-CP2 by executing ``bash tools/download.sh``

Preprocess process the data with bash ``tools/process.sh``

### Training

Run ``python main.py`` to run GenB.


### Evaluating

Run ``python eval.py --load_path DIRNAME`` to evaluate your model. 

For the best performing model, you can download our best performing model from [here](https://drive.google.com/drive/folders/1ujBnfmKHp2m9FDla2zkO-HrBowiLSJtk) and run it.



## Acknowledgements
This repo contains code largely modeified from [here](https://github.com/GeraldHan/GGE).


## Citation
If you find this code useful, please cite our paper:
```
@inproceedings{cho2023genb,
	title={Generative Bias for Robust Visual Question Answering},
	author={Cho, Jae Won and Kim, Dong-Jin and Ryu, Hyeonggon and Kweon, In So},
	booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	year={2023}
}
```

