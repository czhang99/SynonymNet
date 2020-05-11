# Entity Synonym Discovery via Multipiece Bilateral Context Matching

This project provides source code and data for SynonymNet, a model that detects entity synonyms via multipiece bilateral context matching. 

Details about SynonymNet can be accessed [here](https://arxiv.org/abs/1901.00056), and the implementation is based on the Tensorflow library. 

## Quick Links
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Installation

For training, a GPU is recommended to accelerate the training speed. 

### Tensorflow

The code is based on Tensorflow 1.5 and can run on Tensorflow 1.15.0. You can find installation instructions [here](https://www.tensorflow.org/install).

### Dependencies

The code is written in Python 3.7. Its dependencies are summarized in the file ```requirements.txt```. 

tensorflow_gpu==1.15.0<br>
numpy==1.14.0<br>
pandas==0.25.1<br>
gensim==3.8.1<br>
scikit_learn==0.21.2

You can install these dependencies like this:
```
pip3 install -r requirements.txt
```
## Usage
* Run the model on Wikipedia+Freebase dataset with the siamese architecture and the default hyperparameter settings<br>
```cd src```<br>
```python3 train_siamese.py --dataset=wiki```<br>

* For all available hyperparameter settings, use<br>
```python3 train_siamese.py -h```

* Run the model on Wikipedia+Freebase dataset with the triplet architecture and the default hyperparameter settings<br>
```cd src```<br>
```python3 train_triplet.py --dataset=wiki```<br>


## Data
### Format
Data 
Each dataset is a folder under the ```./input_data``` folder, where each sub-folder indicates a train/val/test split:
```
./data
└── wiki
    ├── train
    |   ├── siamese_contexts.txt
    |   └── triple_contexts.txt
    ├── valid
    |   ├── siamese_contexts.txt
    |   └── triple_contexts.txt    
    ├── test
    |   ├── knn-siamese_contexts.txt
    |   ├── knn_triple_contexts.txt
    |   ├── siamese_contexts.txt
    |   └── triple_contexts.txt
    └── skipgram-vec200-mincount5-win5.bin
    └── fasttext-vec200-mincount5-win5.bin
    └── in_vovab (build during training)
```
In each sub-folder,<br> 
* ```siamese_contexts.txt``` file contains entities and contexts for the siamese architecture. Each line has five columns, seperated by \t:
```entity_a \t entity_b \t context_a1@@context_a2...context_an \t context_b1@@context_b2@@...@@context_bn \t  label```.<br>
    * ```entity_a``` and ```entity_b``` indicate two entities. e.g. ```u.s._government||m.01bqks||``` and ```united_states||m.01bqks||```.
    * The next two columns indicate the contexts of two entities. e.g. ```context_a1@@context_a2...context_an``` indicates n pieces of contexts where ```entity_a``` is mentioned. ```@@``` is used to seperate contexts.
    *  ```label``` is a binary value indicating synonymity.
    
* ```triple_contexts.txt``` file contains entities and contexts for the triplet architecture. Each line has six columns, seperated by \t: 
```entity_a \t entity_pos \t entity_neg \t context_a1@@context_a2...context_an \t context_pos_1@@context_pos_2@@...@@context_pos_p \t  context_neg_1@@context_neg_2@@...@@context_neg_q```.<br>
 where ``entity_a`` denotes one entity and ```entity_pos``` denotes a synonym entity of ``entity_a`` and ```entity_neg``` as a negative sample of ``entity_a``.

* ```*-vec200-mincount5-win5.bin``` is a binary file stores the pre-trained word embedding trained using the corpus in the dataset.

* ```in_vocab``` is a vocabulary file generated automatically during training.

### Download
Pre-trained word vectors and datasets can be downloaded here:<br> 

| Dataset  | Link |
| ------------- | ------------- |
| Wikipedia + Freebase  |   |
| PubMed + UMLS  |  |

### Work on your own data
Prepare and organize your dataset in a folder according to the [format](#format) and put it under ```./input_data/``` and use `--dataset=foldername` during training. 

For example, your dataset is `./input_data/mydata`, then you need to use the flag `--dataset=mydata` for ```train_triplet.py```.<br>
Your dataset should be seperated to three folders - train, test, and valid, which is named 'train', 'test', and 'valid' by default setting of ```train_triplet.py``` or ```train_siamese.py```. 
   
## Reference
```
@inproceedings{zhang2020entity,
  title={Entity Synonym Discovery via Multipiece Bilateral Context Matching},
  author={Zhang, Chenwei and Li, Yaliang and Du, Nan and Fan, Wei and Yu, Philip S},
  booktitle={Proceedings of the 29th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2020}
}
```
