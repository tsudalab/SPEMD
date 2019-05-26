import numpy as np
import random
import argparse
import csv
import combo
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import label_propagation
from scipy import stats
import simulator


parser = argparse.ArgumentParser()
parser.add_argument('shape')
parser.add_argument('iteration', type = int)
parser.add_argument('param_dir')
parser.add_argument('threshold', type = float)
parser.add_argument('--test', '-t', default = None)
parser.add_argument('--method', '-m', default = 'BOUS') #BOUS, BO, US, RS
parser.add_argument('--initialRandom', type = int,  default = 10)
args = parser.parse_args()
isTest = args.test
para_shape = [float(v) for v in args.shape.split(',')]
dimension = len(para_shape)
param_dir = args.param_dir
success_threshold = args.threshold
method = args.method
iteration = args.iteration
init_rndm = args.initialRandom



def run(X):
    if method == 'BOUS':
        run_BOUS(X)
    elif method == 'BO':
        run_BO(X)
    elif method == 'US' or method == 'US_simple':
        run_US(X)
    elif method == 'RS':
        run_RS(X)

def run_RS(X):
    
    rndm_para_index_list = list(range(num_param))
    random.shuffle(rndm_para_index_list)

    detected_SP = []
    for it in range(iteration):
        next_param = param_data[rndm_para_index_list[it]]

        #Call simulator
        success_rate = simulator.simulation(next_param, test = isTest)
        
        if success_rate >= success_threshold:
            detected_SP.append(next_param)

        print('iteration:', it+1, 'checked param:', next_param, 'Num. of SPs', len(detected_SP))

def run_BO(X):
    print('BO')    
    
    #Preparation for COMBO
    X = combo.misc.centering( X )
    
    class COMBO_simulator:
        def __init__( self ):
             _ = X
        
        def __call__( self, action ):
            #Call MD simulator
            #print(action, param_data[action[0]])
            success_rate = simulator.simulation(param_data[action[0]], test = isTest)
            return success_rate

    policy = combo.search.discrete.policy(test_X=X)
    res = policy.random_search(max_num_probes=init_rndm, simulator=COMBO_simulator())
    print('checked param:', np.array(param_data)[res.chosed_actions[0:res.total_num_search]], 'Num. of SPs', len([v for v in res.fx[0:res.total_num_search] if v>= success_threshold]))

    for trial in range((iteration-init_rndm)/5):
        res = policy.bayes_search(max_num_probes=5, simulator=COMBO_simulator(), score='EI', interval=5, num_rand_basis=7000)
        best_fx, best_action = res.export_all_sequence_best_fx()
        print('checked param:', np.array(param_data)[res.chosed_actions[res.total_num_search-5:res.total_num_search]], 'Num. of SPs', len([v for v in res.fx[0:res.total_num_search] if v>= success_threshold]))


def run_US(X):
    
    ss = StandardScaler()
    ss.fit(X)
    std_X = ss.transform(X)
    shape = np.array(para_shape)
    shape_ = shape/float(np.max(shape))
    std_X = std_X*shape_
    
    


    rndm_para_index_list = list(range(num_param))
    random.shuffle(rndm_para_index_list)

    detected_SP = []
    detected_rate_list = []
    labeled_index_list = []
    unlabeled_index_list = list(range(num_param))
    phase_list = [-1 for i in range(num_param)]
    for it in range(init_rndm):
        next_param = param_data[rndm_para_index_list[it]]
        #Call simulator
        success_rate = simulator.simulation(next_param, test = isTest)
        if success_rate >= success_threshold:
            detected_SP.append(next_param)

        labeled_index_list.append(rndm_para_index_list[it])
        unlabeled_index_list = [x for x in range(num_param) if x not in labeled_index_list]
        phase_list[rndm_para_index_list[it]] = 1 if success_rate>=success_threshold else 0
        detected_rate_list.append(success_rate)
        print('iteration:', it+1, 'checked param:', next_param, 'Num. of SPs', len(detected_SP))
    
    for it in range(init_rndm, iteration):
        if len(detected_SP) == 0:

            next_param = param_data[rndm_para_index_list[it]]
            #Call simulator
            success_rate = simulator.simulation(next_param, test = isTest)
            if success_rate >= success_threshold:
                detected_SP.append(next_param)
            labeled_index_list.append(rndm_para_index_list[it])
            unlabeled_index_list = [x for x in range(num_param) if x not in labeled_index_list]
            phase_list[rndm_para_index_list[it]] = 1 if success_rate>=success_threshold else 0
            print('iteration:', it+1, 'checked param:', next_param, 'Num. of SPs', len(detected_SP))
        else:
            grid_list = np.array([list(std_X[i])+[phase_list[i]] for i in range(num_param)])
        
            label_train = grid_list[:, -1]
            lp_model = label_propagation.LabelPropagation()
            lp_model.fit(grid_list[:, :-1], label_train)
            predicted_labels = lp_model.transduction_[unlabeled_index_list]
            predicted_all_labels = lp_model.transduction_
            label_distributions = lp_model.label_distributions_[unlabeled_index_list]
            label_distributions_all = lp_model.label_distributions_
            classes = lp_model.classes_

            u_score_list = 1- np.max(label_distributions, axis = 1)
            uncertainty_index = [unlabeled_index_list[np.argmax(1- np.max(label_distributions, axis = 1))]]

            next_param = param_data[uncertainty_index[0]]
            #Call simulator 
            success_rate = simulator.simulation(next_param, test = isTest)
            if success_rate >= success_threshold:
                detected_SP.append(next_param)
        
            labeled_index_list.append(uncertainty_index[0])
            unlabeled_index_list = [x for x in range(num_param) if x not in labeled_index_list]
            phase_list[uncertainty_index[0]] = 1 if success_rate>=success_threshold else 0
            print('iteration:', it+1, 'checked param:', next_param, 'success rate:', success_rate, 'Num. of SPs:', len(detected_SP))

def run_BOUS(X):
    #Preparation for COMBO
    X = combo.misc.centering( X )

    class COMBO_simulator:
        def __init__( self ):
            _ = X

        def __call__( self, action ):
            #Call MD simulator
            success_rate = simulator.simulation(param_data[action[0]], test = isTest)
            return success_rate
    
    #Initial Sampling
    policy = combo.search.discrete.policy(test_X=X)
    res = policy.random_search(max_num_probes=init_rndm, simulator=COMBO_simulator())
    best_fx, best_action = res.export_all_sequence_best_fx()
    print('checked param:', np.array(param_data)[res.chosed_actions[0:res.total_num_search]], 'Num. of SPs', len([v for v in res.fx[0:res.total_num_search] if v>= success_threshold]))

    #BO Sampling
    if best_fx[-1] < success_threshold:
        for trial in range((iteration-init_rndm)/5):
            res = policy.bayes_search(max_num_probes=5, simulator=COMBO_simulator(), score='EI', interval=5, num_rand_basis=7000)
            best_fx, best_action = res.export_all_sequence_best_fx()
            print('checked param:', np.array(param_data)[res.chosed_actions[res.total_num_search-5:res.total_num_search]], 'Num. of SPs', len([v for v in res.fx[0:res.total_num_search] if v>= success_threshold]))
            if best_fx[-1] >= success_threshold:
                break
    
    #US Sampling
    labeled_index_list = list(res.chosed_actions[0:res.total_num_search])
    unlabeled_index_list = [x for x in range(num_param) if x not in labeled_index_list]
    phase_list = [-1 for i in range(num_param)]
    for i in range(res.total_num_search):
        tmp_index = res.chosed_actions[i]
        tmp_rate = res.fx[i]
        phase_list[tmp_index] = 1 if tmp_rate >= success_threshold else 0
    detected_SP = [param_data[i] for i in range(num_param) if phase_list[i] == 1]
    ss = StandardScaler()
    ss.fit(X)
    std_X = ss.transform(X)
    shape = np.array(para_shape)
    shape_ = shape/float(np.max(shape))
    std_X = std_X*shape_

    print('detected SPs:', detected_SP)

    for it in range(len(labeled_index_list), iteration):
        grid_list = np.array([list(std_X[i])+[phase_list[i]] for i in range(num_param)])
        #grid_list = np.array([std_X[i] for i in range(num_param)])

        label_train = grid_list[:, -1]
        lp_model = label_propagation.LabelSpreading()
        lp_model.fit(grid_list, label_train)
        predicted_labels = lp_model.transduction_[unlabeled_index_list]
        predicted_all_labels = lp_model.transduction_
        label_distributions = lp_model.label_distributions_[unlabeled_index_list]
        label_distributions_all = lp_model.label_distributions_
        classes = lp_model.classes_
        
        u_score_list = 1- np.max(label_distributions, axis = 1)
        uncertainty_index = [unlabeled_index_list[np.argmax(1- np.max(label_distributions, axis = 1))]]

        next_param = param_data[uncertainty_index[0]]
        #Call simulator-
        success_rate = simulator.simulation(next_param, test = isTest)
        if success_rate >= success_threshold:
            detected_SP.append(next_param)

        labeled_index_list.append(uncertainty_index[0])
        unlabeled_index_list = [x for x in range(num_param) if x not in labeled_index_list]
        phase_list[uncertainty_index[0]] = 1 if success_rate>=success_threshold else 0
        print('iteration:', it+1, 'checked param:', next_param, 'Num. of SPs', len(detected_SP))
    

if __name__ == '__main__':

    #Load parameter datai
    print('Load parameter list...')
    param_data = []
    
    with open(param_dir, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            param_data.append([float(v) for v in row])
    num_param = len(param_data)

    X = np.array(param_data)[:,:dimension]
    if isTest == None:
        X = np.array(param_data)[:,:dimension]
    elif isTest == 'Newtonian':
        X = np.array(param_data)[:,:dimension]
        X = np.log(X)
    elif isTest == 'Langevin':
        X = np.array(param_data)[:,:dimension]
        X[:,1] = np.log(X[:,1])
   

    #Run parameter optimizer
    run(X)

