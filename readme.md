Нейросеть на основе vgg16, предобученной на imagenet, с обученным классификатором, обучаемая с методами аугментации данных. В файле train_step.py хранится код для нейросети с уменьшением lr ступенчато, в файле train_exp.py -- экспоненциально,в файле train_step_warmup.py -- ступенчато с разогревом, в файле train__exp_warmup.py -- экспоненциально с разогревом.

Ступенчато

```
def step_decay(epoch):    
    initial_lrate = 1e-11
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
```

1) initial_lrate = 1e-10

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/step-10.png)

2) initial_lrate = 1e-11

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/step-11.png)

3) Совмещенные графики

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/step.png)

При темпе в 1e-10 потери изначально немного возрастают, но они остаются ниже, чем при initial_lrate = 1e-11, а также растет точность на валидационных данных, поэтому этот темп на мой взгляд лучше.

Экспоненциально

    def exp_decay(epoch):
        initial_lrate = 1e-11
        k = 0.1 
        lrate = initial_lrate * math.exp(-k*epoch)
        return lrate
1) initial_lrate = 1e-10

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/exp-10.png)

2) initial_lrate = 1e-11

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/exp-11.png)

3) Совмещенные графики

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/exp.png)

Как и в ступенчатом случае, при 1e-10 наблюдается небольшой рост потерь, но они ниже, чем при 1e-11, и точность на валидационных данных растет.

Ступенчато с разогревом

```
def step_decay(epoch):    
    initial_lrate = 1e-11
    drop = 0.5
    epochs_drop = 10.0
    if epoch < 10:
        lrate = initial_lrate/(10-epoch)
    else:
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
```

1) initial_lrate = 1e-10

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/step_warm-10.png)

2) initial_lrate = 1e-11

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/step_warm-11.png)

3) Совмещенные графики

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/step_warm.png)

Как и ранее наблюдается тенденция, что при старте с 1e-10 изначально растут потери, но растет точность на валидационной выборке.

4) Совмещенные графики оптимальных начальных темпов для ступенчатого темпа с разогревом и без

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/step_b.png)

На вилидационных данных значения примерно одинаковые, но потери на обучающей выборке ниже при темпе с разогревом, поэтому этот вариант кажется лучше.

Экспоненциально с разогревом

``` 
def exp_decay(epoch):
    initial_lrate = 1e-11
    k = 0.1
    if epoch < 10:
        lrate = initial_lrate/(10-epoch)
    else:  
        lrate = initial_lrate * math.exp(-k*epoch)
    return lrate
```

1) initial_lrate = 1e-10

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/exp_warm-10.png)

2) initial_lrate = 1e-11

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/exp_warm-11.png)

3) Совмещенные графики

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/exp_warm.png)

Как и ранее, потери изначально растут при 1e-10, но точность на валидации повышается

4) Совмещенные графики оптимальных начальных темпов для ступенчатого темпа с разогревом и без

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/exp_b.png)

В случае экспоненты точность на валидационных данных выше при обучении без разогрева, поэтому предпочтительнее этот способ.

Совмещенные графики лучших ступенчатого темпа с разогревом и экспоненциального без разогрева

![Image alt](https://github.com/samsdimko/SMOMI5/blob/master/b.png)

В этих случаях оба темпа примерно одинаково себя показывают на валидации, но экспоненциальный темп лучше на обучении.

