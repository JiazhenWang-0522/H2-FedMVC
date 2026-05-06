import os
# You can set the parameters you want here by yourself.
Dirichlet_alpha = [9999]
# 'NUSWIDE', 'MNIST-USPS', 'Fashion', 'BDGP'
dataset = ['NUSWIDE', 'MNIST-USPS', 'Fashion', 'BDGP']
# [2, 1, 0.5]
M_S = [2, 1, 0.5]

for alpha in Dirichlet_alpha:
    for ds in dataset:
        for ms in M_S:
            print(f"Running alpha={alpha}, dataset={ds}, M_S={ms}")
            os.system(
                f"python main.py --Dirichlet_alpha {alpha} --dataset {ds} --M_S {ms}"
            )
