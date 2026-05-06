import math
import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import torch
import os
import scipy.io as sio
import metric
from my_network import Network
from torch.utils.data import DataLoader
import numpy as np
import argparse
import random
import copy
from loss import Loss
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from dataloader import load_data, DatasetSplit, get_mask
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

os.environ['OMP_NUM_THREADS'] = '1'
import warnings
# ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# MNIST-USPS
# BDGP
# Fashion
# NUSWIDE
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default='MNIST-USPS')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.6, type=float)
parser.add_argument("--temperature_l", default=0.7, type=float)
parser.add_argument("--learning_rate", default=0.0001, type=float)
parser.add_argument("--MOE_lr", default=0.01, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)

parser.add_argument("--mse_epochs", default=250)  # pre-training rounds
parser.add_argument("--main_epochs", default=25)  # local training rounds
parser.add_argument("--MOE_epochs", default=100)

parser.add_argument("--feature_dim", default=15)  # d_m and d
parser.add_argument("--num_users", default=24)  # number of clients

# This parameter controls the distribution of each category. If you want to set it as non-iid, please set this parameter to 0.5.
parser.add_argument("--Dirichlet_alpha", type=float, default=9999.0)

parser.add_argument("--M_S", type=float, default=2)
parser.add_argument("--interval_epoch", default=10)
parser.add_argument("--participate", default=1)  # client participation rates
parser.add_argument("--tau", default=0.2, type=float, help="Interpolation strength for mutual learning")
parser.add_argument("--thr", default=0.1, type=float, help="Interpolation strength for mutual learning")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)

if args.dataset == "MNIST-USPS":
    args.num_users = 24
    args.main_epochs = 25
    args.interval_epoch = 25
    seed = 10
if args.dataset == "NUSWIDE":
    args.num_users = 24
    args.main_epochs = 25
    args.interval_epoch = 25
    seed = 10
if args.dataset == "BDGP":
    args.num_users = 12
    args.main_epochs = 25
    args.interval_epoch = 25
    seed = 10
if args.dataset == "Fashion":
    args.num_users = 48
    args.main_epochs = 25
    args.interval_epoch = 25
    seed = 10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def pretrain(nu, model):
    model.train()
    mes = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for pre_epoch in range(args.mse_epochs):
        tot_loss = 0.
        for batch_idx, (xs, ys, xs_cluster) in enumerate(data_loader_list[nu]):
            for v in range(view):
                xs[v] = xs[v].to(device)
                xs[v] = xs[v].to(torch.float32)
            optimizer.zero_grad()
            xrs, q, h, rs = model(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(mes(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

    h_list = []
    ys_list = []
    for batch_idx, (xs, ys, xs_cluster) in enumerate(data_loader_list[nu]):
        for v in range(view):
            xs[v] = xs[v].to(device)
            xs[v] = xs[v].to(torch.float32)
        _, _, h, _ = model(xs)
        h_list.append(h)
        ys_list.append(ys)
    h_list = torch.cat(h_list, dim=0)
    ys_list = torch.cat(ys_list, dim=0)
    kmeans = KMeans(n_clusters=class_num, init='k-means++', n_init=100)
    kmeans.fit(h_list.detach().cpu().numpy())
    labels = kmeans.predict(h_list.detach().cpu().numpy())
    print('client', nu, 'pretrain acc', compute_acc(labels, ys_list.detach().cpu().numpy()))
    cluster_centers = kmeans.cluster_centers_
    model.centroids.data = copy.deepcopy(torch.tensor(cluster_centers)).to(device)

    for idx, label in enumerate(labels):
        data_loader_list[nu].dataset.xs_cluster[idx] = cluster_centers[label]

def local_pretrain(nu, model, glob_model, isfull):
    model.train()
    mes = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
    for epoch in range(1):
        tot_loss = 0.
        for batch_idx, (xs, ys, xs_cluster) in enumerate(data_loader_list[nu]):
            for v in range(view):
                xs[v] = xs[v].to(device)
                xs[v] = xs[v].to(torch.float32)
            optimizer.zero_grad()
            xrs, zs, h, rs = model(xs)
            _, glob_z, glob_h, glob_r = glob_model(xs)
            loss_list = []
            for v in range(view):
                if isfull:
                    loss_list.append(criterion.forward_feature(rs[v],h))
                else:
                    if v in num_views[nu]:
                        loss_list.append(criterion.forward_model(glob_h, h, zs[v]))
                loss_list.append(mes(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

def local_train_cluster_center(nu, model, glob_model, isfull):
    model.train()
    mes = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
    for epoch in range(1):
        tot_loss = 0.
        for batch_idx, (xs, ys, xs_cluster) in enumerate(data_loader_list[nu]):
            for v in range(view):
                xs[v] = xs[v].to(device)
                xs[v] = xs[v].to(torch.float32)
            optimizer.zero_grad()
            xrs, zs, h, rs = model(xs)
            _, glob_z, glob_h, glob_r = glob_model(xs)
            loss_list = []
            for v in range(view):
                if isfull:
                    loss_list.append(criterion.forward_feature_cluster_center(rs[v], xs_cluster.to(device)))
                else:
                    if v in num_views[nu]:
                        loss_list.append(criterion.forward_model(glob_h, h, zs[v]))
                loss_list.append(mes(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

def local_full_train(nu, local_model, glob_models_temp):
    glob_models_weights = copy.deepcopy(glob_models_temp)
    for key, model in glob_models_temp.items():
        model.train()
        num_view = list(key)
        mes = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        for epoch in range(50):
            tot_loss = 0.
            for batch_idx, (xs, ys, xs_cluster) in enumerate(data_loader_list[nu]):
                xs_missing = copy.deepcopy(xs)
                for v in range(view):
                    xs[v] = xs[v].to(device)
                    xs[v] = xs[v].to(torch.float32)
                    if v not in num_view:
                        xs_missing[v] = torch.tensor(np.zeros((len(xs_missing[v]), dims[v])))
                    xs_missing[v] = xs_missing[v].to(device)
                    xs_missing[v] = xs_missing[v].to(torch.float32)
                optimizer.zero_grad()
                _, _, h_ref, _ = local_model(xs)
                xrs, zs, h, rs = model(xs_missing)
                if epoch == 0:
                    _, _, glob_h, _ = model(xs_missing)
                loss_list = []
                for v in range(view):
                    loss_list.append(mes(xs[v], xrs[v]))
                loss_list.append(mes(h, h_ref))

                loss = sum(loss_list)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()

        mi = 0
        for v in range(view):
            if v in num_view:
                if num_view.__len__() < view:
                    mi = metric.mutual_information(h, glob_h) - metric.mutual_information(h, zs[num_view[0]])
                else:
                    mi += metric.mutual_information(h, rs[v])

        glob_models_temp[key] = model
        glob_models_weights[key] = mi

    return glob_models_temp, glob_models_weights

def local_single_train(nu, local_model, glob_model, glob_models_temp):
    glob_models_weights = copy.deepcopy(glob_models_temp)
    for key, model in glob_models_temp.items():
        model.eval()
        num_view = list(key)
        mi = 0
        if num_views[nu] == num_view:
            for batch_idx, (xs, ys, xs_cluster) in enumerate(data_loader_list[nu]):
                for v in range(view):
                    xs[v] = xs[v].to(device)
                    xs[v] = xs[v].to(torch.float32)

                xrs, zs, h, rs = local_model(xs)
                _, glob_z, glob_h, glob_r = glob_model(xs)
            mi = metric.mutual_information(h, glob_h) - metric.mutual_information(h, zs[num_view[0]])

            glob_models_temp[key] = local_model
        glob_models_weights[key] = mi

    return glob_models_temp, glob_models_weights

def valid_global(model, valid_dataset_list):
    local_hs, local_ys = [], []
    for an in range(args.num_users):
        h_list, ys_list = [], []
        model.eval()
        for batch_idx, (xs, ys, xs_cluster) in enumerate(valid_dataset_list[an]):
            for v in range(len(xs)):
                xs[v] = xs[v].to(device)
                xs[v] = xs[v].to(torch.float32)
            xrs, _, h, rs = model(xs)
            h_list.append(h)
            ys_list.append(ys)
        local_hs.append(torch.cat(h_list, dim=0))
        local_ys.append(torch.cat(ys_list, dim=0))

    global_hs = torch.cat(local_hs, dim=0)
    global_ys = torch.cat(local_ys, dim=0)

    kmeans = KMeans(n_clusters=class_num, init='k-means++', n_init=100)
    kmeans.fit(global_hs.detach().cpu().numpy())
    labels = kmeans.predict(global_hs.detach().cpu().numpy())
    print('global model acc', compute_acc(labels, global_ys.detach().cpu().numpy()))

def valid_client(model, valid_dataset, nu):
    h_list, ys_list, rs_list = [], [], []
    model.eval()
    for batch_idx, (xs, ys, xs_cluster) in enumerate(valid_dataset):
        for v in range(len(xs)):
            xs[v] = xs[v].to(device)
            xs[v] = xs[v].to(torch.float32)
        xrs, _, h, rs = model(xs)
        rss = []
        for v in range(len(xs)):
            if v in num_views[nu]:
                rss.append(rs[v])
        rss = torch.cat(rss, dim=1)
        rs_list.append(rss)
        h_list.append(h)
        ys_list.append(ys)

    local_hs = torch.cat(h_list, dim=0)
    local_ys = torch.cat(ys_list, dim=0)

    kmeans = KMeans(n_clusters=class_num, init='k-means++', n_init=100)
    kmeans.fit(local_hs.detach().cpu().numpy())
    labels = kmeans.predict(local_hs.detach().cpu().numpy())

    acc = compute_acc(labels, local_ys.detach().cpu().numpy())

    print(f'Client {nu} Accuracy: {acc:.4f}')

    return kmeans.cluster_centers_, acc

def valid(valid_model_list, valid_dataset_list, valid_users):
    local_hs, local_ys, local_labels = [], [], []
    init_center = []
    for an in range(valid_users):
        h_list, ys_list = [], []
        valid_model_list[an].eval()
        for batch_idx, (xs, ys, xs_cluster) in enumerate(valid_dataset_list[an]):
            for v in range(len(xs)):
                xs[v] = xs[v].to(device)
                xs[v] = xs[v].to(torch.float32)
            xrs, _, h, rs = valid_model_list[an](xs)
            h_list.append(h)
            ys_list.append(ys)
        local_hs = torch.cat(h_list, dim=0)
        local_ys.append(torch.cat(ys_list, dim=0))

        if an == 0:
            kmeans = KMeans(n_clusters=class_num, init='k-means++', n_init=100)
            kmeans.fit(local_hs.detach().cpu().numpy())
            init_center = kmeans.cluster_centers_
        else:
            kmeans = KMeans(n_clusters=class_num, init=init_center)
            kmeans.fit(local_hs.detach().cpu().numpy())
        labels = kmeans.predict(local_hs.detach().cpu().numpy())
        local_labels.append(labels)

    global_ys = torch.cat(local_ys, dim=0)
    global_labels = np.concatenate(local_labels, axis=0)

    test_acc = compute_acc(global_labels, global_ys.detach().cpu().numpy())
    test_nmi = normalized_mutual_info_score(global_labels, global_ys.detach().cpu().numpy())
    test_ari = adjusted_rand_score(global_labels, global_ys.detach().cpu().numpy())
    print('overall acc', test_acc)
    print('overall nmi', test_nmi)
    print('overall ari', test_ari)

    return test_acc, test_nmi, test_ari

def match(y_pred, y_true):
    assert y_pred.size == y_true.size
    D = class_num
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):  # 5000
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return ind[1]


def compute_acc(y_pred, y_true):
    assert y_pred.size == y_true.size
    D = total_class_num
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    correct_count = sum([w[i, j] for i, j in zip(ind[0], ind[1])])
    return correct_count * 1.0 / y_pred.size


def aggregate_models(model_list, weights=None):
    if weights is None:
        weights = [1.0 / len(model_list)] * len(model_list)
    agg_model = copy.deepcopy(model_list[0])
    agg_state_dict = agg_model.state_dict()
    for key in agg_state_dict:
        agg_state_dict[key].zero_()
    for model, weight in zip(model_list, weights):
        model_state_dict = model.state_dict()
        for key in agg_state_dict:
            agg_state_dict[key] += model_state_dict[key] * weight
    agg_model.load_state_dict(agg_state_dict)
    return agg_model


def aggregate_models1(model_list):
    # group
    grouped_dict = defaultdict(list)
    for idx, sub_list in enumerate(num_views):
        grouped_dict[tuple(sub_list)].append(idx)
    glob_models = dict()
    for key, value in grouped_dict.items():
        agg_models = []
        for v in value:
            agg_models.append(model_list[v])
        glob_models[key] = aggregate_models(agg_models)
    return grouped_dict, glob_models


def aggregate_models2(model_list, grouped_dict, weights_list):
    glob_models = dict()
    for key, _ in grouped_dict.items():
        agg_models = []
        agg_weights = []
        for nu in range(len(model_list)):
            agg_models.append(model_list[nu][key])
            agg_weights.append(weights_list[nu][key])
        total_sum = np.sum(agg_weights)
        weights = agg_weights / total_sum
        glob_models[key] = aggregate_models(agg_models, weights=weights)
    return glob_models


def save_data(num_users, num_views, data_loader_list, missing_rate):
    len1 = len(data_loader_list[0].dataset.idxs)
    mask = np.ones((len1, view))
    for nu in range(1, num_users):
        len1 = len(data_loader_list[nu].dataset.idxs)
        mask1 = []
        for v in range(view):
            if v in num_views[nu]:
                mask_temp = np.ones((len1, 1))
            else:
                mask_temp = np.zeros((len1, 1))
            if len(mask1) == 0:
                mask1 = mask_temp
            else:
                mask1 = np.hstack((mask1, mask_temp))
        mask = np.vstack((mask, mask1))

    data = {'mask': mask}
    sio.savemat('./mask/' + args.dataset + str(missing_rate) + '.mat', data)

def build_similarity_matrix(client_cluster_centers, device=None, keep_diag_zero=True, use_mean=False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    C = torch.stack([torch.as_tensor(c, dtype=torch.float32, device=device)
                     for c in client_cluster_centers])
    C = F.normalize(C, p=2, dim=-1)
    S_raw = torch.einsum('nkd,mqd->nm', C, C)
    if keep_diag_zero:
        S_raw.fill_diagonal_(0)
    if use_mean:
        S_raw = S_raw / (C.size(1) * C.size(1))
    S_row_squared = S_raw ** 2
    S_row_softmax = F.softmax(S_row_squared, dim=1)


    return S_raw, S_row_softmax

def save_client_data_distribution(dataset, num_users, dataset_name, trial_id):
    filename = f"client_data\\client_data_distribution_{dataset_name}_trial_{trial_id}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Dateset: {dataset_name}\n")
        f.write(f"ID: {trial_id}\n")
        f.write(f"client_nums: {num_users}\n")
        f.write("=" * 50 + "\n\n")

        for client_id in range(num_users):
            client_data_indices = dataset.user_data[client_id]
            client_labels = dataset.Y[client_data_indices]

            # 统计每个类别的样本数量
            unique_labels, counts = np.unique(client_labels, return_counts=True)

            f.write(f"client {client_id}:\n")
            f.write(f"  Total sample size: {len(client_data_indices)}\n")
            f.write(f"  The included categories: {sorted(unique_labels.tolist())}\n")
            f.write(f"  The number of samples in each category:\n")

            for label, count in zip(unique_labels, counts):
                f.write(f"    category {label}: {count} \n")

            f.write("\n")

if __name__ == '__main__':
    T = 5  # repeated experiment
    accs = []
    nmis = []
    aris = []

    for i in range(T):

        missing_mapping = {2: 0.3333, 1: 0.5, 0.5: 0.6667}
        missing_rate = missing_mapping.get(args.M_S)

        # # non-iid
        # if args.dataset == "BDGP":
        #     num_of_client_categories = 3
        # elif args.dataset == "Fashion":
        #     num_of_client_categories = 5
        # elif args.dataset == "MNIST-USPS":
        #     num_of_client_categories = 5
        # elif args.dataset == "NUSWIDE":
        #     num_of_client_categories = 3

        # iid
        if args.dataset == "BDGP":
            num_of_client_categories = 5
        elif args.dataset == "Fashion":
            num_of_client_categories = 10
        elif args.dataset == "MNIST-USPS":
            num_of_client_categories = 10
        elif args.dataset == "NUSWIDE":
            num_of_client_categories = 5


        dataset, dims, view, data_size, class_num, total_class_num = load_data(args.dataset, args.num_users,
                                                                               args.Dirichlet_alpha,
                                                                               num_of_client_categories)
        data_loader_list = []
        test_data_loader_list = []

        num_users = args.num_users
        num_views = get_mask(view, num_users, missing_rate)
        num_views_glob = get_mask(view, num_users, missing_rate)

        for j in range(num_users):
            user_idx = np.array(dataset.user_data[j])
            client_dict = {}

            for v in range(view):
                Xv = dataset.X[v][user_idx]
                if hasattr(Xv, "cpu"):
                    Xv = Xv.cpu().numpy()

                client_dict[f"X_view{v + 1}"] = Xv

            Yj = dataset.Y[user_idx]
            if hasattr(Yj, "cpu"):
                Yj = Yj.cpu().numpy()
            client_dict["Y"] = Yj

            client_dict["indices"] = user_idx
            client_dict["view_mask"] = np.array(num_views_glob[j])

            mat_filename = f"client_data\\{args.dataset}_client_{j + 1}.mat"
            sio.savemat(mat_filename, client_dict)

            data_loader = DataLoader(
                DatasetSplit(dataset.X, dataset.Y, dataset.user_data[j], dims, num_views_glob[j]),
                batch_size=args.batch_size,
                shuffle=False,
            )
            data_loader_list.append(copy.deepcopy(data_loader))

        for j in range(num_users):
            test_data_loader = DataLoader(
                DatasetSplit(dataset.X, dataset.Y, dataset.user_data[j], dims, num_views_glob[j]),
                batch_size=data_size, shuffle=False)
            test_data_loader_list.append(copy.deepcopy(test_data_loader))

        local_models = []

        client_cluster_centers = []

        print('Start Training')

        setup_seed(seed)

        num_users = int(args.num_users * args.participate)
        num_full_client = math.ceil(num_users * (1 - missing_rate))
        num_views = num_views_glob[:num_full_client]
        num_full_client_glob = math.ceil(args.num_users * (1 - missing_rate))
        num_views = num_views + num_views_glob[num_full_client_glob:num_full_client_glob + num_users - num_full_client]

        for j in range(num_users):
            local_models.append(
                copy.deepcopy(Network(view, num_views[j], dims, args.feature_dim, class_num, device).to(device)))

        save_client_data_distribution(dataset, num_users, args.dataset, i)

        global_center_list = []

        for nu in range(num_users):
            pretrain(nu, local_models[nu])
            if nu + 1 < num_users:
                local_models[nu + 1] = local_models[0]

        global_model = aggregate_models(local_models)
        grouped_dict, glob_models = aggregate_models1(local_models)

        print('Start pretraining')
        for round in range(2):
            glob_models_list = []
            glob_models_weights_list = []
            for nu in range(num_users):
                isfull = nu<num_full_client
                glob_models_temp = copy.deepcopy(glob_models)
                for me in range(args.main_epochs):
                    if isfull:
                        local_models[nu] = glob_models[tuple(num_views[nu])]
                    local_pretrain(nu, local_models[nu], glob_models[tuple(num_views[nu])], isfull)
                    if me % args.interval_epoch == 0 or me == 0:
                            valid_client(local_models[nu], data_loader_list[nu],nu)

                if isfull:
                    glob_models_temp, glob_models_weights = local_full_train(nu, local_models[nu], glob_models_temp)
                    glob_models_list.append(glob_models_temp)
                    glob_models_weights_list.append(glob_models_weights)

                else:
                    if round>0:
                        glob_models_temp, glob_models_weights = local_single_train(nu, local_models[nu], glob_models[tuple(num_views[nu])], glob_models_temp)
                        glob_models_list.append(glob_models_temp)
                        glob_models_weights_list.append(glob_models_weights)

        h_avg = [[] for _ in range(num_users)]

        print('Start cluster_training')
        for round in range(5):
            print(f"\nStart the {round + 1}th round of federated learning training.")

            glob_models_list = []
            glob_models_weights_list = []

            client_accuracies = []
            for nu in range(num_users):
                isfull = nu < num_full_client
                glob_models_temp = copy.deepcopy(glob_models)

                for me in range(args.main_epochs):
                    if isfull:
                        local_models[nu] = glob_models[tuple(num_views[nu])]
                    local_train_cluster_center(nu, local_models[nu], glob_models[tuple(num_views[nu])], isfull)
                    if me % args.interval_epoch == 0 or me == 0:
                        cluster_centers, acc = valid_client(local_models[nu], data_loader_list[nu], nu)
                        if nu < len(client_cluster_centers):
                            client_cluster_centers[nu] = cluster_centers
                        else:
                            client_cluster_centers.append(cluster_centers)
                        client_accuracies.append(acc)

                        filename = f"acc\\client_accuracy_{args.temperature_f}_{args.temperature_l}_{args.learning_rate}_{args.tau}.txt"
                        with open(filename, 'a', encoding='utf-8') as f:
                            f.write(f"Round: {round + 1}, Epoch: {me}, Client: {nu}, Accuracy: {acc:.4f}\n")

                    local_models[nu].eval()
                    h_list_epoch = []
                    with torch.no_grad():
                        for batch_idx, (xs, ys, xs_cluster) in enumerate(data_loader_list[nu]):
                            for v in range(view):
                                xs[v] = xs[v].to(device).to(torch.float32)
                            _, _, h_batch, _ = local_models[nu](xs)
                            h_list_epoch.append(h_batch)

                    h_all = torch.cat(h_list_epoch, dim=0)  # [num_samples_client, 15]

                    h_avg[nu].append(h_all.detach().cpu())


                    kmeans_epoch = KMeans(n_clusters=class_num, init='k-means++', n_init=100)
                    kmeans_epoch.fit(h_all.detach().cpu().numpy())
                    labels_epoch = kmeans_epoch.labels_
                    centers_epoch = kmeans_epoch.cluster_centers_

                    local_models[nu].centroids.data = torch.tensor(centers_epoch, dtype=torch.float32, device=device)

                    dataset_nu = data_loader_list[nu].dataset
                    dataset_nu.xs_cluster = centers_epoch[labels_epoch]

                if isfull:
                    glob_models_temp, glob_models_weights = local_full_train(nu, local_models[nu], glob_models_temp)
                    glob_models_list.append(glob_models_temp)
                    glob_models_weights_list.append(glob_models_weights)
                else:
                    if round > 0:
                        glob_models_temp, glob_models_weights = local_single_train(nu, local_models[nu],glob_models[tuple(num_views[nu])],glob_models_temp)
                        glob_models_list.append(glob_models_temp)
                        glob_models_weights_list.append(glob_models_weights)

            # 计算相似度矩阵
            if len(client_cluster_centers) > 1:
                S_raw, S_soft = build_similarity_matrix(client_cluster_centers, device=device,
                                                        keep_diag_zero=True, use_mean=False)

                acc_vec = torch.tensor(client_accuracies, dtype=torch.float32, device=S_soft.device)
                tau = args.tau

                n_clients = S_soft.size(0)
                thr = args.thr

                if args.dataset == 'BDGP':
                    acc_thr = 0.9
                if args.dataset == 'MNIST-USPS':
                    acc_thr = 0.9
                if args.dataset == 'Fashion':
                    acc_thr = 0.8
                if args.dataset == 'NUSWIDE':
                    acc_thr = 0.5


                for i in range(n_clients):
                    if acc_vec[i] >= acc_thr:
                        continue

                    mask_soft = S_soft[i] > thr  # [n]
                    mask_acc = acc_vec > acc_vec[i]  # [n]
                    mask = mask_soft & mask_acc
                    mask[i] = False

                    idx_js = torch.nonzero(mask, as_tuple=False).flatten().tolist()
                    if len(idx_js) == 0:
                        continue


                    w_raw = S_soft[i, idx_js]
                    weights = w_raw / (w_raw.sum() + 1e-9)

                    with torch.no_grad():
                        teacher_params = [torch.zeros_like(p_i.data) for p_i in local_models[i].parameters()]
                        for w, j in zip(weights.tolist(), idx_js):
                            for tp, p_j in zip(teacher_params, local_models[j].parameters()):
                                tp.add_(p_j.data, alpha=float(w))

                        # 插值更新：θ_i ← (1-τ)θ_i + τ·θ_teacher
                        for p_i, tp in zip(local_models[i].parameters(), teacher_params):
                            p_i.data.mul_(1.0 - tau).add_(tp, alpha=tau)

            save_dir = "client_models"
            os.makedirs(save_dir, exist_ok=True)

            model_name = f"{args.dataset}_MS_{args.M_S:.2f}_Dirichlet_alpha_{args.Dirichlet_alpha:.2f}"

            for nu in range(num_users):
                model_path = os.path.join(
                    save_dir,
                    f"client_{nu}_{model_name}.pt"
                )
                torch.save(local_models[nu].state_dict(), model_path)

            print(f"All client models have been saved to the directory.: {save_dir}")

        def _centers_to_cluster_id(xc_np: np.ndarray, decimals: int = 6) -> np.ndarray:
            xc_round = np.round(xc_np.astype(np.float64), decimals=decimals)
            xc_view = np.ascontiguousarray(xc_round).view(
                np.dtype((np.void, xc_round.dtype.itemsize * xc_round.shape[1]))
            )
            _, inv = np.unique(xc_view, return_inverse=True)
            return inv


        def spread_classes_2d(X2: np.ndarray, ids: np.ndarray, alpha: float = 4.0, scale: float = 1.0) -> np.ndarray:
            X2 = X2.astype(np.float64, copy=True)
            ids = ids.astype(np.int64, copy=False)

            uniq = np.unique(ids)
            centers = {}
            for k in uniq:
                centers[k] = X2[ids == k].mean(axis=0)

            mean_center = np.stack([centers[k] for k in uniq], axis=0).mean(axis=0)

            shift = np.zeros_like(X2)
            for k in uniq:
                idx = (ids == k)
                shift[idx] = centers[k] - mean_center

            return scale * X2 + alpha * shift

        load_dir = r"client_models\\"
        print(f"Start loading the client model from {load_dir}...")

        model_name = f"{args.dataset}_MS_{args.M_S:.2f}_Dirichlet_alpha_{args.Dirichlet_alpha:.2f}"

        for nu in range(len(local_models)):
            pt_name = f"client_{nu}_{model_name}.pt"
            model_path = os.path.join(load_dir, pt_name)

            print(f"[Load] Client {nu} <- {pt_name}")

            state = torch.load(model_path, map_location=device)
            local_models[nu].load_state_dict(state)

        print("All client models have been loaded successfully！")

        from fed_moe import FedMoE

        feature_dim = args.feature_dim
        num_classes = args.num_users

        moe_model = FedMoE(
            client_models=local_models,
            feature_dim=feature_dim,
            num_classes=num_classes,
            top_k=6,
            device=device
        ).to(device)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, moe_model.parameters()),
            lr=args.MOE_lr
        )
        criterion_ce = torch.nn.CrossEntropyLoss()

        global_loaders = data_loader_list

        moe_epochs = args.MOE_epochs
        for epoch in range(moe_epochs):
            moe_model.train()
            total_loss = 0.0
            total_cnt = 0

            for nu, loader in enumerate(global_loaders):
                for xs, ys, xs_cluster in loader:
                    for v in range(len(xs)):
                        xs[v] = xs[v].to(device).to(torch.float32)
                    ys = ys.to(device, dtype=torch.long)

                    logits, gate_idx, gate_score, mixed_h = moe_model(xs)
                    loss = criterion_ce(logits, ys)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * ys.size(0)
                    total_cnt += ys.size(0)

            print(f"[MoE] Epoch {epoch + 1}/{moe_epochs}, loss={total_loss / total_cnt:.4f}")

        moe_model.eval()
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        VIS_CLIENT_ID = 0

        client1_feats = []
        client1_true_labels = []
        client1_cluster_ids = []

        global_feats = []
        global_true_labels = []
        global_cluster_ids = []

        SPREAD_ALPHA = 4.0
        SPREAD_SCALE = 1.0

        with torch.no_grad():
            for nu, loader in enumerate(data_loader_list):
                for xs, ys, xs_cluster in loader:
                    for v in range(len(xs)):
                        xs[v] = xs[v].to(device).to(torch.float32)
                    ys = ys.to(device, dtype=torch.long)

                    logits, gate_idx, gate_score, mixed_h = moe_model(xs)

                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == ys).sum().item()
                    total += ys.size(0)

                    all_preds.append(preds.detach().cpu().numpy())
                    all_labels.append(ys.detach().cpu().numpy())

                    # ---------- collect GLOBAL ----------
                    g_feat = mixed_h if mixed_h is not None else logits
                    g_feat = g_feat.detach()
                    if g_feat.dim() > 2:
                        g_feat = g_feat.view(g_feat.size(0), -1)
                    global_feats.append(g_feat.cpu().numpy())
                    global_true_labels.append(ys.detach().cpu().numpy())

                    if torch.is_tensor(xs_cluster):
                        g_xc = xs_cluster.detach().cpu().numpy()
                    else:
                        g_xc = np.asarray(xs_cluster)
                    if g_xc.ndim > 2:
                        g_xc = g_xc.reshape(g_xc.shape[0], -1)
                    global_cluster_ids.append(_centers_to_cluster_id(g_xc, decimals=6))

                    # ---------- collect CLIENT1 ----------
                    if nu == VIS_CLIENT_ID:
                        c_feat = mixed_h if mixed_h is not None else logits
                        c_feat = c_feat.detach()
                        if c_feat.dim() > 2:
                            c_feat = c_feat.view(c_feat.size(0), -1)
                        client1_feats.append(c_feat.cpu().numpy())
                        client1_true_labels.append(ys.detach().cpu().numpy())

                        if torch.is_tensor(xs_cluster):
                            c_xc = xs_cluster.detach().cpu().numpy()
                        else:
                            c_xc = np.asarray(xs_cluster)
                        if c_xc.ndim > 2:
                            c_xc = c_xc.reshape(c_xc.shape[0], -1)
                        client1_cluster_ids.append(_centers_to_cluster_id(c_xc, decimals=6))

        test_acc = correct / max(total, 1)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        moe_nmi = normalized_mutual_info_score(all_labels, all_preds)
        moe_ari = adjusted_rand_score(all_labels, all_preds)

        acc100 = test_acc * 100.0
        nmi100 = moe_nmi * 100.0
        ari100 = moe_ari * 100.0

        print(f"[MoE] ACC(%): {acc100:.2f}")
        print(f"[MoE] NMI(%): {nmi100:.2f}")
        print(f"[MoE] ARI(%): {ari100:.2f}")

        accs.append(acc100)
        nmis.append(nmi100)
        aris.append(ari100)












