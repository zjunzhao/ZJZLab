I reproduce Instance Recognition Algorithm introduced by [\[1\]](https://arxiv.org/abs/1805.01978) (resnet_cifar.py is clone from \[2\]) and modify it to get a slightly better performence with much shorter training time.



The testing results of original Instance Recognition Algorithm and modified Instance Recognition Algorithm on CIFAR-10 dataset are as follows (each model has been trained five times, each time I random select 5000 examples from training set to do validation and use the solution that performs best on validation set to do testing). The performence of models is measured by KNN accuracy on testing data with visual embeddings in memory bank to be training data and visual embeddings calculated from training data to be training data, corresponding to metrics acc_1 and acc_2. The performence of original Instance Recognition Algorithm reproduce by me is slightly worse than the results in [\[1\]](https://arxiv.org/abs/1805.01978) because [\[1\]](https://arxiv.org/abs/1805.01978) uses testing data to do validation.

|model|acc_1|acc_2|training time|
|:----:|:----:|:----:|:----:|
|original IR|![](http://latex.codecogs.com/gif.latex?\\0.791\pm0.002)|![](http://latex.codecogs.com/gif.latex?\\0.786\pm0.002)|![](http://latex.codecogs.com/gif.latex?\\9407\pm1033)|
|modified IR|![](http://latex.codecogs.com/gif.latex?\\0.804\pm0.002)|![](http://latex.codecogs.com/gif.latex?\\0.800\pm0.001)|![](http://latex.codecogs.com/gif.latex?\\4816\pm1162)|

## reference
\[1\][Unsupervised Feature Learning via Non-parametric Instance Discrimination](https://arxiv.org/abs/1805.01978)
\[2\][Codes by authors of paper](https://github.com/zhirongw/lemniscate.pytorch)