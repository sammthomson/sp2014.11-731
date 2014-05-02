Version 1
---------

trained a pairwise ranking classifier with adagrad, perceptron loss

no regularization, 100 iterations

decoded with instant runoff

BLEU:   0.243675174142

METEOR: 0.3274973405550851


Version 2
---------

Added features for:
* number of non-ASCII words
* number of commas
* length of the sentence

added L2 regularization w/ lambda = 1.0

BLEU:   0.249476658131

METEOR: 0.32963374501281356


Version 3
---------

Fixed gradient bug

BLEU:   0.246010974999

METEOR: 0.33063065246156326
