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

# Experiment 2:
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