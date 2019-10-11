I reproduce wide residual network and test WRN-28-10 on CIFAR-10 dataset.

The architecture of WRN-28-10 is as follows (generate by hiddenlayer package).

[WRN-28-10](figures/WRN-28-10.png)

The testing results of WRN-28-10 on CIFAR-10 dataset is as follows (each model has been trained five times, each time I random select 5000 examples from training set to do validation and use the solution that performs best on validation set to do testing).

|model|testing loss|testing acc|
|:----:|:----:|:----:|
|WRN-28-10-nodropout|$0.156\pm0.002$|$0.960\pm0.001$|
|WRN-28-10-0.3dropout|$0.162\pm0.003$|$0.959\pm0.002$|

## reference
[Wide Residual Networks](https://arxiv.org/pdf/1605.07146v1.pdf)