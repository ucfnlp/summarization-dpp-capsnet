# Improving the Similarity Measure of Determinantal Point Processes for Extractive Multi-Document Summarization

We provide the source code for the paper **"[Improving the Similarity Measure of Determinantal Point Processes for Extractive Multi-Document Summarization](https://arxiv.org/abs/1906.00072)"**, accepted at ACL'19. If you find the code useful, please cite the following paper. 

    @inproceedings{cho-lebanoff-foroosh-liu:2019,
     Author = {Sangwoo Cho and Logan Lebanoff and Hassan Foroosh and Fei Liu},
     Title = {Improving the Similarity Measure of Determinantal Point Processes for Extractive Multi-Document Summarization},
     Booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
     Year = {2019}}

This repository contains the code for a similarity measure network using [Capsule network](https://github.com/XifengGuo/CapsNet-Keras).

## Dependencies
This code is developed with the following environment:
- [Python 2.7](https://www.anaconda.com/download/)
- [Keras 2.2.4](https://keras.io/)
- [Tensorflow 1.12.0 backend](https://www.tensorflow.org/install)
- `pip install -r requirements.txt`

## Set up directory for work
```
$ git clone https://github.com/sangwoo3/summarization-dpp-capsnet.git & cd summarization-dpp-capsnet
$ mkdir data & cd data
```

## Train and evaluate on the CNN/DM summary pair dataset
### Download the data
1. Download CNN/DM summary pair dataset from [HERE](https://drive.google.com/file/d/1GyvIHzoTat0xSZsdYJjocLji4eNnQ6NW/view?usp=sharing) and extract it under `/data` directory (This dataset is pre-processed with 50k prevailing vocabularies in CNN/DM and zipped into train, validation, and test set.)
2. Download Glove word vectors of 50k vocabulary from [HERE](https://drive.google.com/file/d/1IpVT7LQ73_yqPYHaHHIAJ2iWPva0x2Uv/view?usp=sharing) and place it under `/data` directory
3. If you look for the raw CNN/DM summary pair dataset, download from [HERE](https://drive.google.com/file/d/1_c4AqnEct0HMg0VOWqupcO0_ijn-fJb0/view?usp=sharing). (This data is pre-processed with the `preprocess.py` file.)

### Training
`$ python main_Capsnet.py`

### Testing
`$ python main_Capsnet.py --testing`

### Testing on STS dataset
`$ python main_Capsnet.py --testing --test_mode STS`

### Pre-trained Model
- Download the model from [HERE](https://drive.google.com/file/d/11-Bc_BhBFRDTUWhM3ihiCm385GGkGWzZ/view?usp=sharing) and place it under `/result/capnet_sim` directory (or your assigned training result folder)
- Download the model fine-tuned on STS dataset from [HERE](https://drive.google.com/file/d/1LR6MncA2ViNVQqcMxxFR6fAYMUaoA6Rs/view?usp=sharing) (This model is used to evaluate STS prediction accuracy.)


### System summary
We provide our best system summaries of DUC04 and TAC11 generated with DPP in the `system_summary` directory.
You can refer to [DPP](https://www.alexkulesza.com/) code and download [DUC 03/04](https://duc.nist.gov/) and [TAC 08/09/10/11](https://tac.nist.gov/data/index.html) dataset with approval.

## License

This project is licensed under the BSD License - see the [LICENSE.md](LICENSE.md) file for details.


