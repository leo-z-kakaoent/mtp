import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def fetch_data():
    fn11 = ['gs://leo_melon/20230614_mtp/dataset_20230614_20221101_000000000%s.parquet'%str(i).zfill(3) for i in range(167)]
    fn12 = ['gs://leo_melon/20230614_mtp/dataset_20230614_20221201_000000000%s.parquet'%str(i).zfill(3) for i in range(167)]
    fn01 = ['gs://leo_melon/20230614_mtp/dataset_20230614_20230101_000000000%s.parquet'%str(i).zfill(3) for i in range(167)]
    fn02 = ['gs://leo_melon/20230614_mtp/dataset_20230614_20230201_000000000%s.parquet'%str(i).zfill(3) for i in range(167)]
    
    train_fn = fn11 + fn12
    valid_fn = fn01
    test_fn = fn02
    
    print("Fetching Train Datasets")
    temp = []
    for f in tqdm(train_fn):
        temp.append(pd.read_parquet(f))
    train_df = pd.concat(temp)
    
    print("Fetching Valid Datasets")
    temp = []
    for f in tqdm(valid_fn):
        temp.append(pd.read_parquet(f))
    valid_df = pd.concat(temp)
    
    print("Fetching Test Datasets")
    temp = []
    for f in tqdm(test_fn):
        temp.append(pd.read_parquet(f))
    test_df = pd.concat(temp)
    
    return train_df, valid_df, test_df


def metric_summary(true_y, pred_y):
    return {
        "confusion_matrix": confusion_matrix(true_y, pred_y),
        "accuracy": accuracy_score(true_y, pred_y),
        "precision": precision_score(true_y, pred_y),
        "recall":recall_score(true_y, pred_y, zero_division=0),
        "f1": f1_score(true_y, pred_y),
    }


class BinarySampler:

    def __init__(self, label, n_batch):
        self.n_batch = n_batch

        label = np.array(label)
        self.indices = []
        for i in [0,1]:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.indices.append(ind)
            
        size1 = len(self.indices[1])-1
        self.batch_size = [int(size1*2), size1]

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            for i in [0,1]:
                alabels = self.indices[i]
                pos = torch.randperm(len(alabels))[:self.batch_size[i]]
                batch.append(alabels[pos])
            batch = torch.concat(batch)
            yield batch


class MyDataset(Dataset):
    def __init__(self, df):
        array_columns = df.columns[[10,16,22]]
        scalar_index = np.delete(np.arange(len(df.columns)),[0,10,16,22,161,162,163])
        scalar_columns = df.columns[scalar_index]
        binary_columns = df.columns[161]
        label_columns = df.columns[162]
        
        scalar_values = df[scalar_columns].values
        array_values = np.stack(df[array_columns].apply(np.concatenate, axis=1).values)
        binary_values = np.stack(df[binary_columns].apply(lambda x: np.array([int(a) for a in list(x)])).values)
        
        x = np.column_stack([scalar_values, array_values, binary_values]).astype(np.float32)
        y = np.eye(2, dtype=np.float32)[df[label_columns].values.astype(np.int32)]
        
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(nn.Module):
    
    def __init__(self, first_size, reduction_scale):
        super().__init__()
        self.dense0 = nn.Linear(217, first_size)
        self.bn0 = nn.BatchNorm1d(first_size)
        self.dense1 = nn.Linear(first_size,int(first_size*reduction_scale))
        self.bn1 = nn.BatchNorm1d(int(first_size*reduction_scale))
        self.dense2 = nn.Linear(int(first_size*reduction_scale),int(first_size*reduction_scale**2))
        self.bn2 = nn.BatchNorm1d(int(first_size*reduction_scale**2))
        self.dense_ce = nn.Linear(int(first_size*reduction_scale**2),2)
        self.dense_triplet = nn.Linear(int(first_size*reduction_scale**2),int(first_size*reduction_scale**3))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.dense0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.bn2(x)
        #
        x_ce = self.softmax(self.dense_ce(x))
        x_triplet = torch.tanh(self.dense_triplet(x))
        return x_ce, x_triplet


def mixed_training(model, optimizer, ce_loss_function, triplet_loss_function, dataloader, bdim, gamma):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        x, y = batch
        ce_output, triplet_output = model(x.cuda())
        
        anchor_output = triplet_output[:bdim]
        positive_output = triplet_output[bdim:(2*bdim)]
        negative_output = triplet_output[(2*bdim):(3*bdim)]
        
        ce_loss = ce_loss_function(ce_output, y.cuda())
        triplet_loss = triplet_loss_function(anchor_output, positive_output, negative_output)
        loss = ce_loss * gamma + triplet_loss
        
        loss.backward()
        optimizer.step()

def inference(model, dataloader):
    metrics = dict()
    model.eval()
    preds = []
    trues = []
    for batch in dataloader:
        x, y = batch
        pred, _ = model(x.cuda())
        preds.append(pred.cpu().detach().numpy().squeeze())
        trues.append(y.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return np.argmax(preds, axis=1), np.argmax(trues, axis=1)

def params_parcer(param_indices, params_dict, param_size=1000):
    # discretization
    discrete_param_indices = [np.clip(p, a_min=0, a_max=param_size).astype(np.int32) for p in list(param_indices)]
    already_exp = tuple(discrete_param_indices) in list(params_dict.keys())
    # index to params
    params = {
        'first_size': np.linspace(128,1024,1000).astype(np.int32)[discrete_param_indices[0]],
        'reduction_scale': np.linspace(0.1,0.9,1000)[discrete_param_indices[1]],
        'learningrate': np.linspace(1e-5,1e-2,1000)[discrete_param_indices[2]],
        'gamma': np.linspace(1e-8,2,1000)[discrete_param_indices[3]],
        'n_epoch': np.linspace(500,5000,1000).astype(np.int32)[discrete_param_indices[4]],
    }
    return tuple(discrete_param_indices), already_exp, params


def aexp(param_indices, train_dl, valid_dl, test_dl, params_dict, tqdm_on=False):
    
    discrete_param_indices, already_exp, params = params_parcer(param_indices, params_dict)
    
    if already_exp:
        return discrete_param_indices, params_dict[discrete_param_indices]
    else:
        first_size = params['first_size']
        reduction_scale = params['reduction_scale']
        learningrate = params['learningrate']
        gamma = params['gamma']
        n_epoch = params['n_epoch']

        bdim = train_dl.batch_sampler.batch_size[1]
        
        torch.manual_seed(22)

        model = MLP(first_size, reduction_scale)
        model = model.cuda()

        loss_function = nn.CrossEntropyLoss()
        triplet_loss_function = nn.TripletMarginLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

        best_f1 = 0
        best_metrics = {}
        for i in tqdm(range(n_epoch)):
            mixed_training(model, optimizer, loss_function, triplet_loss_function, train_dl, bdim, gamma)

            if i % 10 == 0:
                preds, trues = inference(model, valid_dl)
                valid_metrics = metric_summary(true_y=trues, pred_y=preds)
                preds, trues = inference(model, test_dl)
                test_metrics = metric_summary(true_y=trues, pred_y=preds)

                if best_f1 < valid_metrics['f1']:
                    best_metrics['valid'] = valid_metrics
                    best_metrics['test'] = test_metrics

        return discrete_param_indices, best_metrics

def insert_experiment(client, a):
    rows_to_insert = [
        {
            "param_set": str(a[0]), 
            "valid_f1": np.around(a[1]['valid']['f1'], decimals=8),
            "valid_accuracy": np.around(a[1]['valid']['accuracy'],decimals=8),
            "valid_precision": np.around(a[1]['valid']['precision'],decimals=8),
            "valid_recall": np.around(a[1]['valid']['recall'],decimals=8),
            "test_f1": np.around(a[1]['test']['f1'],decimals=8),
            "test_accuracy": np.around(a[1]['test']['accuracy'], decimals=8),
            "test_precision": np.around(a[1]['test']['precision'],decimals=8),
            "test_recall": np.around(a[1]['test']['recall'],decimals=8),
        },
    ]

    errors = client.insert_rows_json("mtp.exp_test", rows_to_insert)  # Make an API request.
    if errors == []:
        print("New rows have been added.")
    else:
        print("Encountered errors while inserting rows: {}".format(errors))


class Simplex:
    def __init__(self, f, train_dl, valid_dl, test_dl, dim=6, gamma=[1,3], rho=[-1,-0.1], sigma=[0.1,1]):
        self.dim = dim
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.f = f
        self.client = bigquery.Client()
        
    def get_simplex(self):
        pass
        
    def sort_simplex(self):
        self.res.sort(key=lambda x: -x[1])
    
    def get_centroid(self):
        x0 = [0.] * self.dim
        for tup in self.res[:-1]:
            for i, c in enumerate(list(tup[0])):
                x0[i] += c / (len(self.res)-1)
        return np.array(x0)
    
    def reflection(self, x0):
        xr = x0 + (x0 - np.array(self.res[-1][0]))
        params, adict = self.f(xr, self.train_dl, self.valid_dl, self.test_dl, self.params_dict)
        params_dict[params] = adict
        rscore = adict['valid']['f1']
        if -self.res[0][1] <= -rscore < -self.res[-2][1]:
            del self.res[-1]
            self.append_res(params, adict)
        return params, adict 
    
    def expansion(self, x0, xr, rdict):
        rscore = rdict['valid']['f1']
        if -rscore < -self.res[0][1]:
            xe = x0 + self.gamma*(x0 - self.res[-1][0])
            params, adict = self.f(xe, self.train_dl, self.valid_dl, self.test_dl, self.params_dict)
            params_dict[params] = adict
            escore = adict['valid']['f1']
            if -escore < -rscore:
                del self.res[-1]
                self.append_res(params, adict)
            else:
                del self.res[-1]
                self.append_res(xr, rdict)
                
    def contraction(self, x0):
        xc = x0 + self.rho*(x0 - self.res[-1][0])
        params, adict = self.f(xc, self.train_dl, self.valid_dl, self.test_dl, self.params_dict)
        params_dict[params] = adict
        cscore = adict['valid']['f1']
        if -cscore < -self.res[-1][1]:
            del self.res[-1]
            self.append_res(params, adict)
    
    def step(self):
        # Sort Simplex
        self.sort_simplex()
        # Get Centroid
        x0 = self.get_centroid()
        # Reflection
        xr, rdict = self.reflection(x0)
        # expansion
        self.expansion(x0, xr, rdict)
        # contraction
        self.contraction(x0)
        # print result
        self.print_simplex()
    
    def append_res(self, x, res_dict):
        self.res.append([
            x, 
            res_dict['valid']['f1'], 
            res_dict['valid']['precision'],
            res_dict['valid']['recall'],
            res_dict['valid']['accuracy'],
            res_dict['test']['f1'], 
            res_dict['test']['precision'],
            res_dict['test']['recall'],
            res_dict['test']['accuracy'],
        ])
        
    def print_simplex(self):
        columns = ['params','v_f1','v_prec','v_rec','v_acc','t_f1','t_prec','t_rec','t_acc']
        print(pd.DataFrame(self.res, columns=columns).sort_values('v_f1',ascending=False).head())
