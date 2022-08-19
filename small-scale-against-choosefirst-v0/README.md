# small scale experiment v0

## experiment info
Train a host network versus `ChooseFirstAgent`. 20 checkpoints. Total of 200k steps.
Details see config yaml file.

## Network structure
Two layer residue networks:

```
Sequential(
  (0): HostFeatureExtractor(
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (1): ReLU()
  (2): ResidualBlock(
    (lin1): Linear(in_features=60, out_features=256, bias=True)
    (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (lin2): Linear(in_features=256, out_features=256, bias=True)
    (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (down_sample): Sequential(
      (0): Linear(in_features=60, out_features=256, bias=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (3): ReLU()
  (4): ResidualBlock(
    (lin1): Linear(in_features=256, out_features=256, bias=True)
    (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (lin2): Linear(in_features=256, out_features=256, bias=True)
    (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (5): Linear(in_features=256, out_features=4, bias=True)
)
```

## Final validation

10000 step match. Count how many separate games there are during the 10000 steps.

```
PolicyHost vs ChooseFirstAgent
 - number of games:326
PolicyHost vs RandomAgent
 - number of games:1654
RandomHost vs ChooseFirstAgent
 - number of games:488
RandomHost vs RandomAgent
 - number of games:1481
AllCoordHost vs ChooseFirstAgent
 - number of games:10
AllCoordHost vs RandomAgent
 - number of games:2139
Zeillinger vs ChooseFirstAgent
 - number of games:719
Zeillinger vs RandomAgent
 - number of games:1898
```

## Pictures

### Loss
[loss](loss.png)
[rhos](rhos.png)

A few layer weight info (first a few and last a few):
[wt1](layer_weight_samples1.png)
[wt2](layer_weight_samples2.png)
[wt3](layer_weight_samples3.png)


