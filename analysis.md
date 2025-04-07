# Template Experiment:
### Heterogeneous GNN:
#### Tests
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 
- Comments: 
### Homogeneous GNN:
#### Tests
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 
- Comments: 

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
#### 1 Layer, No Skip Connections
- Train metrics: loss 1.5664, Acc 0.3738, F1 0.3438
- Valid metrics: loss 1.6831, Acc 0.2444, F1 0.1977
- Comments: Overfitting
#### 1 Layer, Skip Connections
- Train metrics: loss 1.5634, Acc 0.3476, F1 0.3176
- Valid metrics: loss 1.6800, Acc 0.2556, F1 0.1796
- Comments: Overfitting but slightly better?
#### 2 Layer, No Skip Connections
- Train metrics: loss 1.5803, Acc 0.3167, F1 0.2810
- Valid metrics: loss 1.6973, Acc 0.2444, F1 0.1951
- Comments: Overfitting but better than 1 layer
#### 2 Layer, Skip Connections
- Train metrics: loss 1.5804, Acc 0.3429, F1 0.3177
- Valid metrics: loss 1.6808, Acc 0.2667, F1 0.1953
- Comments: Overfitting, but for some reason when I was testing it out, it did wayyy better (see results/ENZYMES-HeteroGNN_GraphConv-128 hidden channels-2 mlp-2 conv-sum intra_aggr-mean inter_aggr-0.5 dropout-use skip True-0.0001 lr-0.0003 maxlr-0.0001 decay-OneCylceLR-Adam-CE Loss/2025-04-06-21:46 (but this used sum for intra_aggr and mean for inter_aggr that's the only difference)). It got Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1 -> 1.5352	1.6804	0.3619	0.3333	0.3428	0.3136. Which is our new well fit best model!
#### 3 Layer, No Skip Connections
- Train metrics: loss 1.7510, Acc 0.2000, F1 0.1270
- Valid metrics: loss 1.7702, Acc 0.1889, F1 0.0994
- Comments: Struggling to learn generalizable info
#### 3 Layer, Skip Connections (epoch 21)
- Train metrics: loss 1.7251, Acc 0.3048, F1 0.3070
- Valid metrics: loss 1.7465, Acc 0.2889, F1 0.2499
- Comments: Maybe some mild overfitting, but it's doing pretty well.
### Homogeneous GNN:
#### 1 Layer, No Skip Connections
- Train metrics: loss 1.7537, Acc 0.2762, F1 0.2725
- Valid metrics: loss 1.7708, Acc 0.2556, F1 0.2207
- Comments: Doing fairly well for training, but not as good as our best HomoGNN so far
#### 1 Layer, Skip Connections
- Train metrics: loss 1.7629, Acc 0.2643, F1 0.2500
- Valid metrics: loss 1.7785, Acc 0.1778, F1 0.1295
- Comments: Really overfit. This is kinda surprising since I though skip connections were meant to partially address overfitting
#### 2 Layer, No Skip Connections
- Train metrics: loss 1.7057, Acc 0.2619, F1 0.2327
- Valid metrics: loss 1.7429, Acc 0.3000, F1 0.2435
- Comments: Not overfitting anymore. Doing pretty well. (Underfitting on Acc and slightly on F1?)
#### 2 Layer, Skip Connections
- Train metrics: loss 1.7495, Acc 0.2667, F1 0.2611
- Valid metrics: loss 1.7656, Acc 0.2778, F1 0.1713
- Comments: Surprisingly worse than without skip connections...
#### 3 Layer, No Skip Connections
- Train metrics: loss , Acc , F1 
- Valid metrics: loss , Acc , F1 
- Comments: Unable to learn generalizable info
#### 3 Layer, Skip Connections (epoch 25)
- Train metrics: loss 1.7806, Acc 0.2333, F1 0.2374
- Valid metrics: loss 1.7880, Acc 0.2333, F1 0.1204
- Comments: Kinda struggles to learn (?). There are points where it's learning 'generalizable info' but it will crash back down after an epoch to two
## Analysis:
- It seems as though 2 layers is doing pretty well for this work
- Skip connections generally are effective for HeteroGNNs, but not for HomoGNNs. For HomoGNNs, the lack of skip connections performed better, which is odd since I thought residual connections were supposed to help reduce overfitting.

<br><br>

# Experiment 6: Varying width
- Kept residual connections for HeteroGNNs but not HomoGNNs
### Heterogeneous GNN:
#### 32 hidden_channels (epoch 28)
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.7433	1.7626	0.2690	0.2000	0.2690	0.1749
- Comments: Slightly overfitting, but doing pretty well for a small model. It does struggle learn generalizable concepts it seems 
#### 64 hidden_channels
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.6858	1.7547	0.2952	0.2222	0.2696	0.1629
- Comments: Overfits in this instance by a fair bit
#### 128 hidden_channels
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.5802	1.6807	0.3405	0.2667	0.3160	0.1953
- Comments: Also overfits by a fair bit 
#### 256 hidden_channels
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.3519	1.5997	0.4690	0.3333	0.4528	0.3076
- Comments: Still some large overfitting, but the model seems to be doing well at getting higher validation accuracy and F1, which is nice to see
#### 512 hidden_channels
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.1558	1.5232	0.5881	0.3889	0.5858	0.3865
- Comments: Though the models are overfitting here, these are the best validation acc and F1s we've seen so far, which seems kinda promising considering we can save most of these gains with more attuned hyperparameters
### Homogeneous GNN: (no skip)
#### 32 hidden_channels
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- Comments:  Unable to learn generalizable info
#### 64 hidden_channels (epoch 25)
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.7831	1.7907	0.1929	0.1889	0.1359	0.0887
- Comments: Seems like the model at this size is performing relatively poorly and overfits still 
#### 128 hidden_channels
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.7056	1.7433	0.2595	0.3111	0.2307	0.2458
- Comments: This is kinda what we've already seen. Our model hyperparams (lr, decay, etc) are built for this size
#### 256 hidden_channels
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.6927	1.7111	0.2357	0.2111	0.1682	0.1493
- Comments: This is interesting. The model appears to perform exceptionally worse than 128 hidden_channels for some reason
#### 512 hidden_channels
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.6221	1.6941	0.2929	0.2333	0.2576	0.1945
- Comments: The model is beginning to overfit a bit here, but I believe better hyperparams would help with that
### Homogeneous GNN: (skip)
#### 32 hidden_channels
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- Comments:  Unable to learn generalizable info
#### 64 hidden_channels (epoch 25)
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.7781	1.7859	0.2357	0.2222	0.2225	0.1362
- Comments: Seems like the model at this size is performing better than w/o skips and overfits F1 still, but not Acc
#### 128 hidden_channels
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.7497	1.7656	0.2690	0.2778	0.2638	0.1713
- Comments: This is moderately worse than what we saw w/o skips...
#### 256 hidden_channels
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.6786	1.7397	0.3095	0.3000	0.2883	0.2649
- Comments: The training dynamics look good on this one and it doesn't appear to be overfitting. It's slightly behind the HeteroGNN at this size, but the HeteroGNN is overfit
#### 512 hidden_channels (epoch 27)
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.5280	1.6497	0.3810	0.3000	0.3562	0.2837
- Comments: the model is overfitting and didn't seem much gains on 256 w/ skip. I does do much better than 512 w/o skip which is nice, but it doesn't do better than 512 for HeteroGNN w/ skip.
## Analysis:
- Overall, HeteroGNN seems to be doing better than HomoGNN even though HeteroGNN is overfitting.
- Skip connections seems to stabilize training performance and results in better performance than without skips in HomoGNNs.
- HeteroGNN seems to train quite stably, but HomoGNN appears to lose training stability at 512 even with skips.
- Overfitting increases with model size, but validation gains in HeteroGNNs imply better inherent task alignment.

<br><br>

# Experiment 7: Varying aggregation methods
- We'll move forward with 256 hidden channels for both HeteroGNN and HomoGNN to ensure both still can maintain stability and we'll investigate different aggregation methods (and combinations).
- (intra_aggr, inter_aggr) is how they are labeled below
### Heterogeneous GNN:
#### ('sum', 'sum')
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.2642	1.5844	0.5381	0.4000	0.5343	0.3823
- Comments: This is quite overfit, but this is the highest validation acc and F1 we've seen so far! This is better than HeteroGNN exp2 dropout 0.0!
#### ('sum', 'mean')
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.3190	1.6045	0.4810	0.4000	0.4748	0.3787
- Comments: This is even less overfit than the above and maintains a lot of the validation acc and F1. This is the current best model
#### ('sum', 'max')
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.3109	1.5975	0.5310	0.4000	0.5261	0.3774
- Comments: I suppose this is more like sum,sum where it has larger overfitting but still overall lower F1 than sum,mean.
#### ('mean', 'sum')
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.3515	1.5994	0.4690	0.3333	0.4505	0.3076
- Comments: This is what I believe we've been using the whole time, but it's among the worst performing variants.
#### ('mean', 'mean')
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.4121	1.6170	0.4476	0.3444	0.4333	0.3238
- Comments: Training seems to have much higher fluctuations in the valid metrics and still has a fair bit of overfitting
#### ('mean', 'max')
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.4022	1.6203	0.4548	0.2889	0.4407	0.2609
- Comments: This seems to have the same issue as mean,mean where the validaiton matrics seems to fluctuate during training, but including max here and with sum (sum,max) seems to cause validation metrics to be worse overall.
#### ('max', 'sum')
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.3018	1.5988	0.5071	0.3000	0.4993	0.2702
- Comments: The training is more stable, but the overfitting is massive!
#### ('max', 'mean')
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.3568	1.6135	0.4643	0.3444	0.4624	0.3181
- Comments: The overfitting isn't as bad here, but at this point it seems like having mean for inter_aggr is leading to the best performance.
#### ('max', 'max')
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.3518	1.6146	0.4643	0.3444	0.4580	0.3299
- Comments: Interestingly, max,max does better than mean,max. but the training stability seems worse.
### Homogeneous GNN:
#### 'sum'
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.6159	1.6941	0.3357	0.2556	0.3235	0.2221
- Comments: Kinda overfitting here, and the validation metrics aren't much better than what we've seen before
#### 'mean'
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.6777	1.7371	0.3143	0.3000	0.2921	0.2649
- Comments: This one is clearly the best, hardly overfitting (probably cuz we're tuned for this) but still weaker than the HeteroGNN
#### 'max'
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.6679	1.7419	0.3262	0.2333	0.3154	0.1697
- Comments: This one is clearly the worst for HomoGNN. Training seems less stable though
## Analysis:
- Overall, it seems intra_aggr = sum and inter_aggr = mean is our least overfitting model with the best validation scores so far for HeteroGNN (even inter_aggr = sum is great too), and aggr = mean appears to be the best performing aggregation method for our HomoGNN so far but this is what we already had and so HeteroGNN has pulled a fair lead.
- It seems that using max as an aggregation function for either made the validation metrics fluctuate much more, possibly indicating that they aren't learning generalizable info very effectively

<br><br>

# Experiment 8: Specific tuned training
### Heterogeneous GNN:
#### Reduced hidden channels by half and tuned LR
hidden_channels = 128
num_classes = 6
num_layers = 2
intra_aggr='sum'
inter_aggr='mean'
dropout = 0.5
use_skip_connections = True
batch_size = 32
epochs = 45
decay = 2.5e-4
lr = 5e-4
maxlr = 5e-4
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.2842	1.5734	0.4905	0.4333	0.4899	0.4342
- Comments: Not crazy overfit or anything like that, but good
#### Reduce batch size by half
hidden_channels = 256
num_classes = 6
num_layers = 2
intra_aggr='sum'
inter_aggr='mean'
dropout = 0.5
use_skip_connections = True
batch_size = 16
epochs = 45
decay = 5e-4
lr = 5e-4
maxlr = 5e-4
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 0.7022	1.2917	0.7500	0.5667	0.7496	0.5614
- Comments: Wow
#### Reduce batch size by half again
hidden_channels = 256
num_classes = 6
num_layers = 2
intra_aggr='sum'
inter_aggr='mean'
dropout = 0.5
use_skip_connections = True
batch_size = 8
epochs = 45
decay = 5e-4
lr = 5e-4
maxlr = 5e-4
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 0.5562	1.3644	0.7762	0.6333	0.7749	0.6272
- Comments: pretty surprising result
#### Dropout 0.6 and weight decay 2.5e-4
hidden_channels = 256
num_classes = 6
num_layers = 2
intra_aggr='sum'
inter_aggr='mean'
dropout = 0.6
use_skip_connections = True
batch_size = 8
epochs = 45
decay = 2.5e-4
lr = 5e-4
maxlr = 5e-4
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 0.7291	1.3383	0.7238	0.6000	0.7223	0.5916
- Comments: I think we'll call this our final model for HeteroGNN since the generalization gap isn't as large as others and the validation metrics are reasonable with a reasonable validation loss as well
### Homogeneous GNN:
#### Dropped batch size by half
hidden_channels = 256
num_classes = 6
num_layers = 2
aggr = 'mean'
dropout = 0.5
use_skip_connections = True
batch_size = 16
epochs = 45
decay = 5e-4
lr = 1e-4
maxlr = 3e-4
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 1.4088	1.5975	0.4381	0.3444	0.4411	0.3446
- Comments: Dropping batch size seemed to help a lot
#### Dropped batch size by half again
hidden_channels = 256
num_classes = 6
num_layers = 2
aggr = 'mean'
dropout = 0.0
use_skip_connections = True
batch_size = 8
epochs = 45
decay = 5e-4
lr = 1e-4
maxlr = 3e-4
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 0.3432	1.6425	0.8952	0.6000	0.8950	0.6009
- Comments: Dropping batch size and dropout lead to massive overfitting, but MASSIVE validation metric gains (still doesnt beat HeteroGNN with this batch size tho). Actually, it seems like the loss is still unreasonably high so maybe not the best
#### Brought up dropout to 0.2
hidden_channels = 256
num_classes = 6
num_layers = 2
aggr = 'mean'
dropout = 0.2
use_skip_connections = True
batch_size = 8
epochs = 45
decay = 5e-4
lr = 3e-4
maxlr = 3e-4
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 0.8360	1.4862	0.6643	0.5111	0.6647	0.5130
- Comments: 
#### Brought up dropout to 0.4
hidden_channels = 256
num_classes = 6
num_layers = 2
aggr = 'mean'
dropout = 0.4
use_skip_connections = True
batch_size = 8
epochs = 45
decay = 5e-4
lr = 3e-4
maxlr = 3e-4
- Train Loss	Valid Loss	Train Acc	Valid Acc	Train F1	Valid F1
- 	1.1160	1.4342	0.5881	0.5444	0.5845	0.5468
- Comments: It's not really overfitting here, this is kinda cool. I tried different seeds and it seems like it continues to overfit with valid metrics around 0.45 acc, 0.44 F1. I still think this is the best HomoGNN. I think we'll call this our final HomoGNN model

<br><br>

# Experiment 9: Running Best Models on Test Set
### Heterogeneous GNN:
#### Best HeteroGNN:
hidden_channels = 256
num_classes = 6
num_layers = 2
intra_aggr='sum'
inter_aggr='mean'
dropout = 0.6
use_skip_connections = True
batch_size = 8
epochs = 45
decay = 2.5e-4
lr = 5e-4
maxlr = 5e-4

seed 40:
- Train Loss	Valid Loss	Test Loss	Train Acc	Valid Acc	Test Acc	Train F1	Valid F1	Test F1
- 0.7089	1.4444	1.2436	0.7595	0.5778	0.6333	0.7599	0.5817	0.6332
- Comments: 
seed 41:
- Train Loss	Valid Loss	Test Loss	Train Acc	Valid Acc	Test Acc	Train F1	Valid F1	Test F1
- 0.7488	1.2559	1.1877	0.7405	0.5778	0.6222	0.7402	0.5757	0.6254
- Comments: 
seed 42:
- Train Loss	Valid Loss	Test Loss	Train Acc	Valid Acc	Test Acc	Train F1	Valid F1	Test F1
- 0.7157	1.2980	1.6717	0.7381	0.5667	0.5111	0.7357	0.5625	0.5151
- Comments: Possible sensitivity to initialization
### Homogeneous GNN:
#### Best HomoGNN:
hidden_channels = 256
num_classes = 6
num_layers = 2
aggr = 'mean'
dropout = 0.4
use_skip_connections = True
batch_size = 8
epochs = 45
decay = 5e-4
lr = 3e-4
maxlr = 3e-4

seed 40:
- Train Loss	Valid Loss	Test Loss	Train Acc	Valid Acc	Test Acc	Train F1	Valid F1	Test F1
- 1.2149	1.5605	1.3818	0.5286	0.4444	0.5111	0.5145	0.4391	0.5016
- Comments: 
seed 41:
- Train Loss	Valid Loss	Test Loss	Train Acc	Valid Acc	Test Acc	Train F1	Valid F1	Test F1
- 1.1606	1.4489	1.4709	0.5810	0.4556	0.4222	0.5801	0.4593	0.4360
- Comments: 
seed 42:
- Train Loss	Valid Loss	Test Loss	Train Acc	Valid Acc	Test Acc	Train F1	Valid F1	Test F1
- 1.1095	1.4374	1.5998	0.5929	0.5333	0.4556	0.5897	0.5350	0.4506
- Comments:
## Analysis:
- I varied the seeds to see what the training stability is like across different training runs.
Test Acc and F1:
- HeteroGNN achieves higher average test accuracy (58.9%) and F1 (59.1%) compared to HomoGNN (46.3% accuracy, 46.2% F1), demonstrating its superiority on this task. However, HeteroGNN exhibits variability (e.g., seed 42 drops to 51.1% accuracy), suggesting sensitivity to initialization. 
- HomoGNN struggles to match performance, likely due to its inability to model heterogeneous graph structures explicitly
Loss Metrics:
-HeteroGNN generally achieves lower test losses (1.24–1.67 vs. HomoGNN’s 1.38–1.60), except for seed 42 (1.67), aligning with its accuracy drop. This indicates better generalization for HeteroGNN in most cases but highlights instability in certain seeds.
Training Dynamics:
- HeteroGNN shows higher training accuracy (74–76%) vs. HomoGNN (53–59%), suggesting better learning capacity. Despite HeteroGNN’s higher dropout (0.6 vs. 0.4), it avoids severe overfitting, likely due to effective regularization and architectural advantages.
Overall:
- HeteroGNN outperforms the HomoGNN overall, validating its design for heterogeneous graph data. However, its sensitivity to initialization warrants further investigation to ensure robustness. The HomoGNN’s inferior performance underscores the importance of modeling heterogeneity explicitly in such tasks. Future work should focus on stabilizing HeteroGNN training and exploring hybrid architectures.