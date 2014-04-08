There are three Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./decode | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model


Experiments
===========

1
---

Increase beam size to 5

logprob: -5577.706183


2
---

Implemented "skip k" decoding, as in Section 3.1 "IBM Constraints" of Zens, Richard et al.
"Reordering Constraints for Phrase-Based Statistical Machine Translation".
The decoder is allowed to keep a stash of up to `k` skipped phrases as it goes.
At each step, the decoder can either
1. translate an upcoming phrase and append it to the running translation.
2. translate an upcoming phrase and stash it if the stash isn't full.
3. take a phrase out of its stash and append it to the running translation.
We keep a separate beam for each `(num_tokens_processed, num_phrases_skipped)` pair,
where `num_tokens_processed` refers to either translated or skipped.

Note that when `k >= n` and `beam_size = infinity`, this is an exhaustive search.
In this dataset, the maximum sentence length is 29.
Ran with --max-skips=20, --beam-size=100

logprob: -5076.058658
