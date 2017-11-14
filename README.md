# MPRC
Implementation of the model described in the following paper:

- [R^3: Reinforced Ranker-Reader for Open-Domain Question Answering](https://arxiv.org/abs/1709.00023) by Shuohang Wang, Mo Yu, Xiaoxiao Guo, etc..

The Reinforced Ranker-Reader is tested on several open-domain QA datasets: Quasart-T, SearchQA and Triviaqa (unfiltered); The Single Reader is tested on the benchmark data set SQuAD.

### Requirements
- [Torch7](https://github.com/torch/torch7) (with cutorch, cunn, cudnn)
- Python 2.7
- NLTK (with corpus punkt and stopwords)

### Datasets
- [Quasar-T: Datasets for Question Answering by Search and Reading](https://github.com/bdhingra/quasar)
- [SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine](https://github.com/nyu-dl/SearchQA)
- [TriviaQA (unfiltered): A Large Scale Dataset for Reading Comprehension and Question Answering](http://nlp.cs.washington.edu/triviaqa/)
- [SQuAD: Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/data/glove.840B.300d.zip)

### Usage
```
sh preprocess.sh quasart (searchqa/unftriviaqa/squad)
cd main
th main.lua -task quasart (searchqa/unftriviaqa) -model rankerReader -reward_epoch 12
th main.lua -task squad -model mlstmReader 
```

`sh preprocess.sh quasart (searchqa/unftriviaqa)` will download the datasets and preprocess the datasets into the files 
(train.tsv dev.tsv test.tsv) under the path "data/quasart/sequence" with the format:
```
question *split_sign* question_id *split_sign* answer_1 *split_answer* answer_2 ... \n
passage_1 *split_sign* IR_score \n
passage_2 *split_sign* IR_score \n
...
passage_n *split_sign* IR_score \n
*new_instance* \n
question_2 *split_sign* question_id *split_sign* answer_1 *split_answer* answer_2 ... \n
...
```
`main.lua` will first initialize the preprossed data and word embeddings into a Torch format and 
then run the alogrithm. The parameter "-reward_epoch" specifies the epoch when the REINFORCE methond starts to be applied.

`sh preprocess.sh squad` will download the SQuAD dataset and preprocess the it into the files (train.txt dev.txt) under the path "data/squad/sequence" with the format:
```
Passage \t Question \t sequence of the positions where the answer appear in Passage (e.g. 3 4 5 6) \n
```

### Docker
You may try to use Docker for running the code.
- [Nvidia-docker Install](https://github.com/NVIDIA/nvidia-docker)
- [Image](https://hub.docker.com/r/shuohang/mprc/): docker pull shuohang/mprc:1.0

After installation, run the following codes: (**Note**: the repository path "/PATH/mprc" need to change; a task name "quasart or searchqa or unftriviaqa or squad" need to specify)
```
nvidia-docker run -it -v /PATH/mprc:/opt --rm -w /opt      shuohang/mprc:1.0 /bin/bash -c "sh preprocess.sh quasart (searchqa/unftriviaqa/squad)"
nvidia-docker run -it -v /PATH/mprc:/opt --rm -w /opt/main shuohang/mprc:1.0 /bin/bash -c "th main.lua -task quasart (searchqa/unftriviaqa) -model rankerReader -reward_epoch 12"
nvidia-docker run -it -v /PATH/mprc:/opt --rm -w /opt/main shuohang/mprc:1.0 /bin/bash -c "th main.lua -task squad -model mlstmReader"
```
