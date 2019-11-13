I implement two adversarial attack methods: FGSM and PGD, and I use them to do adversarial training. The following are some experiment results. The network i used is WRN-28-10.

||vanilla|FGSM|PGD|
|:---:|:----:|:----:|:----:|
|vanilla|95.3|41.5|0.0|
|FGSM|84.0|94.2|0.0|
|PGD|83.7|59.3|55.0|

The accuracy in ith row, jth column is the test accuracy when attacked by jth column's method, and the model is trained adversarially by ith row's method.
