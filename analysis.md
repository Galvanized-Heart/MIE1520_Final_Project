# Template Experiment:
### Heterogeneous GNN:
#### Tests
- Train metrics: loss , Acc , F1 
- Valid metrics: loss , Acc , F1 
- Comments: 
### Homogeneous GNN:
#### Tests
- Train metrics: loss , Acc , F1 
- Valid metrics: loss , Acc , F1 
- Comments: 

Remaining Experiments:
- Width
- Depth
- Aggr
- Residual connections

<br><br>

# Experiment 1: Testing Different Convs for Het and Hom Architectures
### Heterogeneous GNN:
Adam optimizer

OneCycle learning rate scheduler

hidden_channels = 128

num_classes = 6

num_layers = 2

intra_aggr='mean'

inter_aggr='sum'

dropout = 0.5

batch_size = 32

epochs = 30

lr = 1e-4

maxlr = 3e-4
#### GraphConv: 
- Train metrics: loss 1.5473, Acc 0.3452, F1 0.3148
- Valid metrics: loss 1.6603, Acc 0.3000, F1 0.2669
- Comments: The model is overfitting a little bit, but it seems to do relatively well on this task. The discreptancy between train and valid metrics are relatively moderate as well (0.04 for Acc and F1).
#### SAGEConv:
- Train metrics: loss 1.5515, Acc 0.3476, F1 0.3174
- Valid metrics: loss 1.6639, Acc 0.3000, F1 0.2672
- Comments: Overfitting a little as well, but does extremely slightly better than GraphConv in terms of F1 score. 
#### GATConv:
- Train metrics: loss 1.6725, Acc 0.3024, F1 0.2398
- Valid metrics: loss 1.8050, Acc 0.2000, F1 0.1433
- Comments: Very bad overfitting. I really struggled to get GAT to work well here. (0.10 difference for Acc and 0.09 difference for F1).
### Homogeneous GNN:
Adam optimizer

OneCycle learning rate scheduler

hidden_channels = 128

num_classes = 6

num_layers = 2

aggr='mean'

dropout = 0.5

batch_size = 32

epochs = 30

lr = 1e-4

maxlr = 3e-4
#### GraphConv:
- Train metrics: loss 1.7032, Acc 0.2738, F1 0.2539
- Valid metrics: loss 1.7363, Acc 0.3000, F1 0.2195
- Comments: This model overfits as well, but strangely the valid Acc is 0.03 above and F1 is 0.04 below...
#### SAGEConv:
- Train metrics: loss 1.7044, Acc 0.2833, F1 0.2588
- Valid metrics: loss 1.7360, Acc 0.2333, F1 0.1724
- Comments: Overfitting, 0.05 below on Acc, 0.08 below on F1.
#### GATConv:
- Train metrics: loss 1.7602, Acc 0.2381, F1 0.1508
- Valid metrics: loss 1.7957, Acc 0.1667, F1 0.0633
- Comments: Drastic overfitting, 0.07 below on Acc, 0.09 below on F1. Also struggled to get GAT to work well...
## Analysis:
- Overall, these models are appearing to overfit a bit on the data. But even from this we can see that even our worst HeteroGNN (GATConv) does better than our best HomoGNN (GraphConv). 
- Between the Homo and Hetero GraphConv models, there is still a ~0.04 difference in Acc and F1 score. 
- The difference in size between these two models is quite small. Though there are more different message passing functions in the HeteroGNN and there are slightly more parameters in the classifier, there is considerable gains by processing the data heterogeneously throuhg a HeteroGNN (-0.08 val loss, +0.04 avg train/val acc, +0.05 val F1)

<br><br>

# Experiment 2: Testing Regularization Techniques
- Proceed with GraphConv to keep things comparable
- Try increased/decreased dropout rates

### Heterogeneous GNN:
#### Dropout 0.0
- Train metrics: loss 1.2016, Acc 0.5357, F1 0.5256
- Valid metrics: loss 1.6009, Acc 0.3556, F1 0.3672
- Comments: mega overfitting
#### Dropout 0.25
- Train metrics: loss 1.4016, Acc 0.4381, F1 0.4229
- Valid metrics: loss 1.6258, Acc 0.2889, F1 0.2757
- Comments: mega overfitting
#### Dropout 0.5
- Train metrics: loss 1.5513, Acc 0.3452, F1 0.3164
- Valid metrics: loss 1.6636, Acc 0.3000, F1 0.2672
- Comments: overfitting
#### Dropout 0.6
- Train metrics: loss 1.6533, Acc 0.3048, F1 0.2727
- Valid metrics: loss 1.7269, Acc 0.2111, F1 0.1398
- Comments: overfitting
#### Dropout 0.7
- Train metrics: loss 1.7162, Acc 0.2857, F1 0.2669
- Valid metrics: loss 1.7410, Acc 0.2333, F1 0.1511
- Comments: much better fitting, though still slightly over fitting (ig kinda comparable to 0.5 for acc, but worse F1 score)
#### Dropout 0.8
- Train metrics: loss 1.7590, Acc 0.2429, F1 0.2012
- Valid metrics: loss 1.7818, Acc 0.1889, F1 0.0916
- Comments: begins struggling to learn generalizable info. validation metrics get stuck in unchaging regions for many epochs
#### Dropout 0.9
- Train metrics: loss 1.7752, Acc 0.1833, F1 0.1798
- Valid metrics: loss 1.7906, Acc 0.1667, F1 0.0476
- Comments: unable to derive generalizable info. validation metrics are completely unchanging
### Homogeneous GNN:
#### Dropout 0.0
- Train metrics: loss 1.6423, Acc 0.3524, F1 0.3230
- Valid metrics: loss 1.7194, Acc 0.2333, F1 0.2113
- Comments: mega overfitting
#### Dropout 0.25 (epoch 22)
- Train metrics: loss 1.6903, Acc 0.2524, F1 0.2055
- Valid metrics: loss 1.7418, Acc 0.2000, F1 0.1249
- Comments: overfitting but noticeably lower validation metrics compared to het GNN
#### Dropout 0.5 (epoch 26)
- Train metrics: loss 1.7071, Acc 0.2405, F1 0.2221
- Valid metrics: loss 1.7364, Acc 0.2333, F1 0.1652
- Comments: overfitting but noticeably lower validation metrics compared to het GNN
#### Dropout 0.6 (epoch 20)
- Train metrics: loss 1.7849, Acc 0.2119, F1 0.1573
- Valid metrics: loss 1.7863, Acc 0.2000, F1 0.1244
- Comments: Not very overfit at this point, but the model is beginning to struggle to learn generalizable infor since validation metrics get stuck in unchaging regions for many epochs
#### Dropout 0.7 (epoch 27)
- Train metrics: loss 1.7885, Acc 0.2095, F1 0.1971
- Valid metrics: loss 1.7907, Acc 0.2111, F1 0.1121
- Comments: slightly overfit, but similar to 0.6, struggling to learn generalizable info.
#### Dropout 0.8
- Train metrics: loss , Acc , F1 
- Valid metrics: loss , Acc , F1 
- Comments: unable to derive generalizable info. validation metrics are completely unchanging
#### Dropout 0.9
- Train metrics: loss , Acc , F1 
- Valid metrics: loss , Acc , F1 
- Comments: unable to derive generalizable info. validation metrics are completely unchanging
## Analysis:
- Overall, it seems like 0.5 dropout was a good choice for both models.
- HeteroGNN seems to still perform better with many different dropout percentages.
- HomoGNN seems to struggle to learn generalizable information at a much lower dropout rate than the HeteroGNN. (Is this indicative?)

<br><br>

# Experiment 3: Adding weight decay to optimizer
### Heterogeneous GNN:
#### Weight Decay 1e-4
- Train metrics: loss 1.5595, Acc 0.3405, F1 0.3064
- Valid metrics: loss 1.6726, Acc 0.2556, F1 0.2065
- Comments: similar loss to no decay, but validation metrics have dropped a fair bit (is this more overfitting?)
#### Weight Decay 5e-4
- Train metrics: loss 1.6203, Acc 0.3238, F1 0.2774
- Valid metrics: loss 1.7163, Acc 0.2333, F1 0.1645
- Comments:  decreases in all metrics train/valid
#### Weight Decay 1e-3 (epoch 26)
- Train metrics: loss 1.7007, Acc 0.2524, F1 0.1846
- Valid metrics: loss 1.7618, Acc 0.2333, F1 0.1803
- Comments: This is approching our earlier homoGNN performance but without overfitting (interesting)
### Homogeneous GNN:
#### Weight Decay 1e-4
- Train metrics: loss 1.7061, Acc 0.2619, F1 0.2308
- Valid metrics: loss 1.7429, Acc 0.2889, F1 0.2328
- Comments: Not overfitting (in fact maybe underfitting by acc). This performs better than best HeteroGNN weight decay not overfitting (1e-3)
#### Weight Decay 5e-4
- Train metrics: loss 1.7918, Acc 0.1667, F1 0.0477
- Valid metrics: loss 1.7919, Acc 0.1667, F1 0.0476
- Comments: Unable to learn generalizable info
#### Weight Decay 1e-3
- Train metrics: loss , Acc , F1 
- Valid metrics: loss , Acc , F1 
- Comments: same as 5e-4... not generalizable
## Analysis:
- At this point, we see our first win for HomoGNN; it out performs all the weight decay varieties for HeteroGNN
- Without overfitting at 1e-3, HeteroGNN seems to have similar vaildation metrics to be better than HomoGNN w/o weight decay at dropout 0.5 (exp 2), but this still isnt better than homoGNN.
- (Im curious to see what it would be like for lower dropout rates and include weight decay because HeteroGNN at dropout 0.0 and 0.25 had very high validation metrics overall, but were overfitting!!!)

<br><br>

# Experiment 4: Weight decay + low dropouts (0, 0.25)
- Try using weight decay 1e-4 for these low dropout rates (0.0, 0.25)
### Heterogeneous GNN:
#### Weight decay 1e-4 and Dropout 0.0 (epoch 22)
- Train metrics: loss 1.3747, Acc 0.4524, F1 0.4352 
- Valid metrics: loss 1.6515, Acc 0.2889, F1 0.2866
- Comments: still overfitting
#### Weight decay 1e-4 and Dropout 0.25 (epoch 25)
- Train metrics: loss 1.4491, Acc 0.4071, F1 0.3914
- Valid metrics: loss 1.6520, Acc 0.2889, F1 0.2725
- Comments: still overfitting
#### Weight decay 5e-4 and Dropout 0.25
- Train metrics: loss 1.5379, Acc 0.3714, F1 0.3393
- Valid metrics: loss 1.6828, Acc 0.2778, F1 0.2498
- Comments: still overfitting
#### Weight decay 5e-4 and Dropout 0.0
- Train metrics: loss 1.4797, Acc 0.3833, F1 0.3531
- Valid metrics: loss 1.6802, Acc 0.2667, F1 0.2238
- Comments: still overfitting
### Homogeneous GNN:
#### Weight decay 1e-4 and Dropout 0.0 (epoch 26)
- Train metrics: loss 1.6607, Acc 0.3143, F1 0.2533
- Valid metrics: loss 1.7300, Acc 0.2444, F1 0.1887
- Comments: still overfitting
#### Weight decay 1e-4 and Dropout 0.25 (epoch 26)
- Train metrics: loss 1.6827, Acc 0.2810, F1 0.2311
- Valid metrics: loss 1.7470, Acc 0.2667, F1 0.2121
- Comments: Not as bad ovefitting. About as comparable to HomoGNN weight decay 1e-4 dropout 0.5
#### Weight decay 5e-4 and Dropout 0.25
- Train metrics: loss , Acc , F1 
- Valid metrics: loss , Acc , F1 
- Comments: Unable to learn generalizable info
#### Weight decay 5e-4 and Dropout 0.0 (epoch 26)
- Train metrics: loss 1.6916, Acc 0.2976, F1 0.2431
- Valid metrics: loss 1.6916, Acc 0.3111, F1 0.2538
- Comments: This is surprising since this is doing really well now for HomoGNN
## Analysis:
- No improvements for HeteroGNN, they all continued to overfit.
- HomoGNN dropout 0.25 weight decay 1e-4 was comparable to version that had same weight decay and dropout 0.5.
- HomoGNN is becoming a very strong contender with weight decay 5e-4 dropout 0.0 (this is our best so far!!)

<br><br>

# Experiment 5: Adding skip connections and varying depth
- Let's keep dropout at 0.5 for now to keep things still relatively comparable and we can move to 0.25 for homoGNN if things change.
### Heterogeneous GNN:
#### Tests
- Train metrics: loss , Acc , F1 
- Valid metrics: loss , Acc , F1 
- Comments: 
### Homogeneous GNN:
#### Tests
- Train metrics: loss , Acc , F1 
- Valid metrics: loss , Acc , F1 
- Comments: 

<br><br>