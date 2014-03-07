Experiments
===========

1
---

Features:

* token_precision
* token_recall
* token_f1
* prefix_precision
* prefix_recall
* prefix_f1
* length difference,
* absolute length difference,
* len_factor (len(hyp) / len(ref)),

Tau: 0.144163136735

(no simple morphological meteor, that hurt performance)

2
---

Add:

* character ngrams for n = 3, 4, 5

Tau: 0.15280996921

3
---

Take out everything but token_recall

Tau: 0.16579950585

4
___

Try lots of things unsuccessfully

* smoothed f1
* stemming, looking at p, r, f1, stemmed_meteor
* adding in quadratic feature for length factor (so model could learn optimal factor, and to penalize deviations from it)
