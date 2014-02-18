There are three Python programs here (`-h` for usage):

 - `./align` aligns words using Dice's coefficient.
 - `./check` checks for out-of-bounds alignment points.
 - `./grade` computes alignment error rate.

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./align -t 0.9 -n 1000 | ./check | ./grade -n 5


The `data/` directory contains a fragment of the German/English Europarl corpus.

 - `data/dev-test-train.de-en` is the German/English parallel data to be aligned. The first 150 sentences are for development; the next 150 is a blind set you will be evaluated on; and the remainder of the file is unannotated parallel data.

 - `data/dev.align` contains 150 manual alignments corresponding to the first 150 sentences of the parallel corpus. When you run `./check` these are used to compute the alignment error rate. You may use these in any way you choose. The notation `i-j` means the word at position *i* (0-indexed) in the German sentence is aligned to the word at position *j* in the English sentence; the notation `i?j` means they are "probably" aligned.


Experiments
===========

1
---

Fixed Dice coefficient calculation (intersection was being double counted).
Instead of choosing all alignments that meet a threshold, choose best target word given source word.
Tuned Dice cutoff threshold (best results at -t 0.1).

> Precision = 0.532476

> Recall = 0.565891

> AER = 0.451784


2
---

Model1 estimated with EM, with poor-man's fast align.
Prune log probabilities below -5.0.
Subtract 2.0 * | i/m - j/n | from log probabilities before renormalizing
to penalize off-diagonals.

> Precision = 0.669845

> Recall = 0.661381

> AER = 0.334241
