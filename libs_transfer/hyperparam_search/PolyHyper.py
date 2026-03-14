import random
from scipy.optimize import curve_fit
import torch
import itertools
import numpy as np
from statistics import mean
from scipy.optimize import minimize
from datetime import datetime

# x_data = np.array([[1, 2,.001], [2, 3,.002], [3, 4,.003], [4, 5,.004], [5, 6,.005]])
# y_data = np.array([1.2, 0.5, 0.1, -0.3, -0.8]).reshape(-1, 1)

class Config:
    def __init__(self, data, type, name):
        self.data = data
        self.type = type
        if self.type == 'choice':
            self.idx = {str(d):i for i,d in enumerate(data)}
            self.idx_r = {i:d for i,d in enumerate(data)}
        else:
            self.idx = None
            self.idx_r = None
        self.name = name
        """
        TYPES:
        "choice" -> e.g [1,2,3] discreet parameters like batch_size
        "continuous" -> e.g [1e-1, 1e-3] continuous parameters like learning rate
        """


def PolyHyper_search(create_model, *configs, max_epoch=15, tolerance=0.2, initial_search=5, n_probes=10):
    now = datetime.now()
    file_name = 'hyperparam_optim_' + now.strftime("%d%m%Y") + '_' + now.strftime("%H%M%S")
    now = now.strftime("%d/%m/%Y")+' '+now.strftime("%H:%M:%S")

    names = [config.name for config in configs]
    grid_search = []

    for i, config in enumerate(configs):
        if config.type == 'continuous':
            vals = np.linspace(config.data[0], config.data[1], initial_search).tolist()
            random.shuffle(vals)
            grid_search.append(vals)
        elif config.type == 'choice':
            vals = [random.choice(config.data) for i in range(initial_search)]
            grid_search.append(vals)
        else:
            raise ValueError('improper Config type, types available: choice ; continuous')

    losses=[]
    params = []
    config_db = []
    initial_config=[]
    print('initating initial grid search')
    for i in range(initial_search):
        vals = [v[i] for v in grid_search]
        config = {name: val for name,val in zip(names, vals)}
        vals = [v if c.idx==None else c.idx[str(v)] for v,c in zip(vals,configs)]
        params.append(vals)
        config_db.append(config)
        initial_config.append(config)
        l_l=[]
        model = create_model(config)
        print(f'\nsearch: [{i+1} / {initial_search}] ; config: {config}')
        for epoch in range(max_epoch):
            loss = model.train()
            print(f'\tepoch: {epoch+1}/{max_epoch} loss: {loss:.4f}')
            l_l.append(loss)
        losses.append(l_l)

    end_loss = [[l[-1]] for l in losses]
    min_end = min([l[-1] for l in losses])

    with open(f'Hypers\\{file_name}.txt', 'w') as f:
        f.write(f'Hyperparameter optimization report: {now}\n\nlosses over epochs for configs in initial search:\n')
        for l in losses:
            l = ' '.join(str(round(value,4)) for value in l)
            f.write(str(f'\n{i} config\t{l}'))
        f.write(f'\nconfigs in initial search:')
        for i, c in enumerate(initial_config):
            f.write(str(f'\n{i}\t{c}'))

    config_db_=[]
    losses_=[]
    for i in range(n_probes):
        best_hypers = find_best_hyperparams(np.array(params), np.array(end_loss))
        best_hypers = [v if c.idx == None else find_closest(c.idx.values(), v) for v, c in zip(best_hypers, configs)]
        # print(best_hypers)
        params.append(best_hypers)
        best_hypers = [v if c.idx == None else c.data[v] for v, c in zip(best_hypers, configs)]
        config = {name: v for name, v in zip(names, best_hypers)}
        # print(config, best_hypers)
        # best_hypers = [v if c.idx == None else c.idx[str(v)] for v, c in zip(best_hypers, configs)]
        print(f'\nsearch: [{i + 1} / {n_probes}] ; config: {config}')
        config_db_.append(config)
        config_db.append(config)
        viable_approach = True
        l_l=[]
        l_=0
        model = create_model(config)
        for epoch in range(max_epoch):
            if viable_approach:
                loss = model.train()
                print(f'\tepoch: {epoch + 1}/{max_epoch} loss: {loss:.4f}')
                l_l.append(loss)
                l = (loss + l_)/2
                l_ = loss
                if len(l_l) > 2:
                    if  l > min_end*(1+tolerance) or has_increasing_triplet(l_l):
                        viable_approach = False
                        print(f'ave loss = {l} ; min loss at initial search = {min_end}')
        end_loss.append([loss])
        losses_.append(loss)

        with open(f'Hypers\\{file_name}.txt', 'w') as f:
            f.write(f'Hyperparameter optimization report: {now}\n\nlosses over epochs for configs in initial search:\n')
            for i,l in enumerate(losses):
                l = ' '.join(str(round(value, 4)) for value in l)
                f.write(str(f'\n{i} config loss:\t{l}'))
            f.write(f'\n\nConfigs in initial search:')
            for i, c in enumerate(initial_config):
                f.write(str(f'\n{i}\t{c}'))

            f.write(f'\n\nFinal losses for probes')
            for i,l in enumerate(losses_):
                f.write(str(f'\n{i} config loss:\t{l}'))
            f.write(f'\n\nConfigs in probes:')
            for i, c in enumerate(config_db_):
                f.write(str(f'\n{i}\t{c}'))

    min_loss = min(end_loss)
    min_idx = end_loss.index(min_loss)
    best_config = config_db[min_idx]
    print(f'best config: {best_config} ; loss: {min_loss[0]}')

    with open(f'Hypers\\{file_name}.txt', 'a') as f:
        f.write(f'\n\nbest config: {best_config} ; loss: {min_loss[0]}')

def has_increasing_triplet(lst):
    for i in range(len(lst) - 2):
        if lst[i] < lst[i + 1] < lst[i + 2]:
            return True
    return False

def find_closest(lst, val):
    if not any(isinstance(x, bool) for x in lst):
        return min(lst, key=lambda x:abs(x-val))
    else:
        val = round(val)
        return val == 1

# def fit_log(x):
#     y = [i for i in range(len(x))]
#     def log_func(x, a, b):
#         return a * np.log(x) + b
#
#     popt, pcov = curve_fit(log_func, x, y)
#     a = log_func(50, *popt)
#     if a < 0:
#         a = 0
#     return a

def find_best_hyperparams(x,y,lr=0.005,epochs=5000):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    fitted_coeffs = train_polyfit(x,y,lr,epochs)

    def polynomial_function(x):
        x = np.array([x])
        x = torch.tensor(x, dtype=torch.float32)
        X_poly_test = multivariate_polynomial_features(x)
        X_poly_test = X_poly_test.numpy()
        result = np.dot(X_poly_test, fitted_coeffs).flatten()
        if result[0] < 0:
            result[0] = 0
        return result[0]

    bnds=[] #get the boundaries for hyperparams to be between 0 and the max values
    for i in range(x.shape[1]):
        b = torch.max(x[:,i]).item()
        b_ = torch.min(x[:,i]).item()
        bnds.append((b_, b))
    bnds = tuple(bnds)

    x0 = np.array([(i[0] + i[1])/2 for i in bnds])
    res = minimize(polynomial_function, x0, method="Nelder-Mead", bounds=bnds)
    # print("Optimal solution:", res.x)
    # print("Minimum value:", res.fun)

    return res.x

def multivariate_polynomial_features(x, degree=2):
    n_samples, n_features = x.shape
    terms = []
    for powers in itertools.product(range(degree + 1), repeat=n_features):
        if sum(powers) <= degree:
            terms.append(torch.prod(x ** torch.tensor(powers), dim=1, keepdim=True))
    return torch.cat(terms, dim=1)

def train_polyfit(x,y,lr,epochs):
    X_poly = multivariate_polynomial_features(x)
    weights = torch.randn((X_poly.shape[1], 1), requires_grad=True)
    optimizer = torch.optim.SGD([weights], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.8, step_size=100)
    loss_fn = torch.nn.MSELoss()

    for i in range(1000):
        y_pred = X_poly.mm(weights)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    if loss.item() > 10 or np.isnan(loss.item()):
        lr /= 10
        weights = torch.randn((X_poly.shape[1], 1), requires_grad=True)
        optimizer = torch.optim.SGD([weights], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.8, step_size=100)
        loss_fn = torch.nn.MSELoss()
        for i in range(150):
            y_pred = X_poly.mm(weights)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        if loss.item() > 10 or np.isnan(loss.item()):
            lr *= 100
            weights = torch.randn((X_poly.shape[1], 1), requires_grad=True)
            optimizer = torch.optim.SGD([weights], lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.8, step_size=100)
            loss_fn = torch.nn.MSELoss()
            for i in range(150):
                y_pred = X_poly.mm(weights)
                loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            if loss.item() > 10 or np.isnan(loss.item()):
                print('loss seems too high, possibly adjusting learning rate should help')

    weights = torch.randn((X_poly.shape[1], 1), requires_grad=True)
    optimizer = torch.optim.SGD([weights], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.6, step_size=500)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epochs):
        y_pred = X_poly.mm(weights)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # if epoch % 500 == 0: #for debugging
        #     print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
    print(f'\nloss of the polynominal fit: {loss.item():.6f}')
    return weights.detach().numpy().flatten() #fitted cooefficients