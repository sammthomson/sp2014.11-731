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
