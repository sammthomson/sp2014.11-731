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
---

Try lots of things unsuccessfully

* smoothed f1
* stemming, looking at p, r, f1, stemmed_meteor
* adding in quadratic feature for length factor (so model could learn optimal factor, and to penalize deviations from it)

5
---

Try lots of things unsuccessfully

* p,r,f1 for token ngrams for n = 1, ..., 6
* p,r,f1 for all {1-6}grams together
* p,r,f1 for stemmed token ngrams for n = 1, ..., 6
* p,r,f1 for all stemmed {1-6}grams together
* normalize all vowels to ""
* normalize all vowels to "a"
* score under KenLM, trained on europarl, with and without stemming:
* score / sentence length
* exp(score)

6
---

Finally an improvement!

* strip accents using unidecode
* still use only the one feature: token_recall

Tau: 0.170502
