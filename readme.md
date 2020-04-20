Нейросеть на основе vgg16, предобученной на imagenet, с обученным классификатором, обучаемая с методами аугментации данных. В файле train_step.py хранится код для нейросети с уменьшением lr ступенчато, в файле train_exp.py -- экспоненциально,в файле train_step_warmup.py -- ступенчато с разогревом, в файле train__exp_warmup.py -- экспоненциально с разогревом.

Ступенчато

1) initial_lrate = 1e-10

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/step-10.png)

2) initial_lrate = 1e-11

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/step-11.png)

3) Совмещенные графики

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/step.png)

Экспоненциально

1) initial_lrate = 1e-10

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/exp-10.png)

2) initial_lrate = 1e-11

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/exp-11.png)

3) Совмещенные графики

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/exp.png)

Ступенчато с разогревом

1) initial_lrate = 1e-10

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/step_warm-10.png)

2) initial_lrate = 1e-11

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/step_warm-11.png)

3) Совмещенные графики

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/step_warm.png)

Экспоненциально с разогревом

1) initial_lrate = 1e-10

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/exp_warm-10.png)

2) initial_lrate = 1e-11

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/exp_warm-11.png)

3) Совмещенные графики

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/exp_warm.png)





