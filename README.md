# PyTorch-based Fully Tensorized Recurrent Neural Network Grammars (RNNGs)

## Dependencies
Most of the code should work with PyTorch 1.5 or above. However, if you use mixed-precision training (with `--amp` option), you need to install nightly-version of PyTorch (`pytorch-nightly`), which resolved some tricly bugs around AMP with version `1.6`.

## Data
We first convert a PTB-style dataset (examples found in `data/` folder) into a single json file by preprocessing, which converts each parse tree into tokens, ids (after unkifying), oracle actions, etc:
```
$ python preprocess.py --trainfile data/train.txt --valfile data/valid.txt --testfile data/test.txt 
--outputfile data/ptb --vocabminfreq 1 --unkmethod berkeleyrule
```
By `--unkmethod berkeleyrule`, a unknown token is convert to a special symbol that exploits some surface features, such as `<unk-ly>`, which indidates an unknown token ending with `ly` suffix (e.g., nominally).

The outputs are `data/ptb-train.json`, `data/ptb-val.json`, and `data/ptb-test.json`. Vocabulary is defined by tokens in the training data. Which tokens to include in the vocabulary is decided by `--vocabminfreq` or `--vocabsize` argument (see `python preprocess.py --help`).

## Training
The current implementation is hard-coded for running on a single GPU.

Example to train a model:
```
$ python train.py --train_file data/ptb-train.json --val_file data/ptb-val.json --save_path rnng.pt
--fixed_stack --strategy top_down --dropout 0.3 --optimizer adam --lr 0.01 --gpu 0
```

### Training Tips

- I recomment to use `adam` optimizer instead of `sgd`. Although the original paper reports the results with SGD, I found that Adam works much more stable for this implementation.
- `--lr` and `--dropout` have a large impact on the performance and should be tuned on each dataset.
- Training becomes faster by a larger batch size (e.g., `--batch_size 128`) but the impact to final performance is not fully investigated. For modest amount of data (e.g., English PTB, ~1M tokens), `--batch_size 64` seems to work well.
- `--fixed_stack` is always recommended. Without this, the training will be done by the older code that is not fully tensorized and thus slow.

### Parsing strategies

The above example specifies `--strategy top_down`, which means a parse tree is predicted by completely top-down. In addition to this, we also provide in-order strategy (`--strategy in_order`), which is almost the same as the left-corner strategy in [Kuncoro et al. (2018)](https://www.aclweb.org/anthology/P18-1132/).

## Evaluation
We provide word-synchronous beam search and [particle filter](https://www.aclweb.org/anthology/D19-1106/) for search method at test time. These allows to obtain 1-best parse as well as perplexity as a fully-incremental language model.

Running word-synchronous beam search:
```
$ head -n 3 test.tokens
Influential members of the House Ways and Means Committee introduced legislation that would restrict how the new savings-and-loan bailout agency can raise capital , creating another potential obstacle to the government 's sale of sick thrifts .
The bill , whose backers include Chairman Dan Rostenkowski -LRB- D. , Ill. -RRB- , would prevent the Resolution Trust Corp. from raising temporary working capital by having an RTC-owned bank or thrift issue debt that would n't be counted on the federal budget .
The bill intends to restrict the RTC to Treasury borrowings only , unless the agency receives specific congressional authorization .

$ python beam_search.py --test_file valid.tokens --model_file rnng.pt --batch_size 20 --beam_size 100
--word_beam_size 10 --shift_size 1 --block_size 1000 --gpu 0 --lm_output_file surprisals.txt > valid.parsed
```

`--beam_size`, `--word_beam_size`, and `--shift_size` are three sizes specifying the behavior of word synchronous beam search, corresponding to beam size for transitions between two tokens, beam size saved at each word boundary, and number of beam actions forcefully shifted at each step (i.e., fast-track).

`--block_size` is the number of sentences in a pool, from which mini-baches of sentences are created. Output is dumped after processing every this many sentences.

Alternately, you can search with particle filtering rather than fixed size beam search:
```
$ python beam_search.py --test_file valid.tokens --model_file rnng.pt --batch_size 20 --particle_filter
--particle_size 1000 --gpu 0 --lm_output_file surprisals.txt > valid.parsed
```
