import numpy as np
import pickle
import argparse
import ConfigSpace as CS
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.ERROR)
from copy import deepcopy

import hpbandster
import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker

from pprint import pprint


def create_config_space():
    cs = CS.ConfigurationSpace()

    Adam_final_lr_fraction = CS.UniformFloatHyperparameter('Adam_final_lr_fraction',
                                       lower=1e-4,
                                       upper=1.0,
                                       default_value=1e-2,
                                       log=True)
    cs.add_hyperparameter(Adam_final_lr_fraction)
    Adam_initial_lr = CS.UniformFloatHyperparameter('Adam_initial_lr',
                                                    lower=1e-4,
                                                    upper=1e-2,
                                                    default_value=1e-3,
                                                    log=True)
    cs.add_hyperparameter(Adam_initial_lr)
    SGD_final_lr_fraction = CS.UniformFloatHyperparameter('SGD_final_lr_fraction',
                                                          lower=1e-4,
                                                          upper=1.0,
                                                          default_value=1e-2,
                                                          log=True)
    cs.add_hyperparameter(SGD_final_lr_fraction)
    SGD_initial_lr = CS.UniformFloatHyperparameter('SGD_initial_lr',
                                                   lower=1e-3,
                                                   upper=0.5,
                                                   default_value=1e-1,
                                                   log=True)
    cs.add_hyperparameter(SGD_initial_lr)
    SGD_momentum = CS.UniformFloatHyperparameter('SGD_momentum',
                                                 lower=0,
                                                 upper=0.99,
                                                 default_value=0.9,
                                                 log=False)
    cs.add_hyperparameter(SGD_momentum)
    StepDecay_epochs_per_step = CS.UniformIntegerHyperparameter('StepDecay_epochs_per_step',
                                                         lower=1,
                                                         upper=128,
                                                         default_value=16)
    cs.add_hyperparameter(StepDecay_epochs_per_step)
    activation = CS.CategoricalHyperparameter('activation',
                                              ['relu', 'tanh'],
                                              default_value='relu')
    cs.add_hyperparameter(activation)
                                
    batch_size = CS.UniformIntegerHyperparameter('batch_size',
                                          lower=8,
                                          upper=256,
                                          default_value=16)
    cs.add_hyperparameter(batch_size)
    dropout_0 = CS.UniformFloatHyperparameter('dropout_0',
                                              lower=0.0,
                                              upper=0.5,
                                              default_value=0.0,
                                              log=False)
    cs.add_hyperparameter(dropout_0)
    dropout_1 = CS.UniformFloatHyperparameter('dropout_1',
                                              lower=0.0,
                                              upper=0.5,
                                              default_value=0.0,
                                              log=False)
    cs.add_hyperparameter(dropout_1)
    dropout_2 = CS.UniformFloatHyperparameter('dropout_2',
                                              lower=0.0,
                                              upper=0.5,
                                              default_value=0.0,
                                              log=False)
    cs.add_hyperparameter(dropout_2)
    dropout_3 = CS.UniformFloatHyperparameter('dropout_3',
                                              lower=0.0,
                                              upper=0.5,
                                              default_value=0.0,
                                              log=False)
    cs.add_hyperparameter(dropout_3)
                
    l2_reg_0 = CS.UniformFloatHyperparameter('l2_reg_0',
                                             lower=1e-6,
                                             upper=1e-2,
                                             default_value=1e-4,
                                             log=True)
    cs.add_hyperparameter(l2_reg_0)
    l2_reg_1 = CS.UniformFloatHyperparameter('l2_reg_1',
                                             lower=1e-6,
                                             upper=1e-2,
                                             default_value=1e-4,
                                             log=True)
    cs.add_hyperparameter(l2_reg_1)
    l2_reg_2 = CS.UniformFloatHyperparameter('l2_reg_2',
                                             lower=1e-6,
                                             upper=1e-2,
                                             default_value=1e-4,
                                             log=True)
    cs.add_hyperparameter(l2_reg_2)
    l2_reg_3 = CS.UniformFloatHyperparameter('l2_reg_3',
                                             lower=1e-6,
                                             upper=1e-2,
                                             default_value=1e-4,
                                             log=True)
    cs.add_hyperparameter(l2_reg_3)

    learning_rate_schedule = CS.CategoricalHyperparameter('learning_rate_schedule',
                                                          ['ExponentialDecay', 'StepDecay'],
                                                          default_value='ExponentialDecay')
    cs.add_hyperparameter(learning_rate_schedule)
    loss_function = CS.CategoricalHyperparameter('loss_function',
                                                 ['categorical_crossentropy'],
                                                 default_value='categorical_crossentropy')
    cs.add_hyperparameter(loss_function)
    num_layers = CS.UniformIntegerHyperparameter('num_layers',
                                          lower=1,
                                          upper=4,
                                          default_value=2)
    cs.add_hyperparameter(num_layers)
    num_units_0 = CS.UniformIntegerHyperparameter('num_units_0',
                                           lower=16,
                                           upper=256,
                                           default_value=32)
    cs.add_hyperparameter(num_units_0)
    num_units_1 = CS.UniformIntegerHyperparameter('num_units_1',
                                           lower=16,
                                           upper=256,
                                           default_value=32)
    cs.add_hyperparameter(num_units_1)
    num_units_2 = CS.UniformIntegerHyperparameter('num_units_2',
                                           lower=16,
                                           upper=256,
                                           default_value=32)
    cs.add_hyperparameter(num_units_2)
    num_units_3 = CS.UniformIntegerHyperparameter('num_units_3',
                                           lower=16,
                                           upper=256,
                                           default_value=32)
    cs.add_hyperparameter(num_units_3)
    optimizer = CS.CategoricalHyperparameter('optimizer',
                                             ['Adam', 'SGD'],
                                             default_value='Adam')
    cs.add_hyperparameter(optimizer)
    output_activation = CS.CategoricalHyperparameter('output_activation',
                                                     ['softmax'],
                                                     default_value='softmax')
    cs.add_hyperparameter(output_activation)

    # add conditions

    cond = CS.EqualsCondition(Adam_final_lr_fraction, optimizer, 'Adam')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(Adam_initial_lr, optimizer, 'Adam')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(SGD_momentum, optimizer, 'SGD')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(SGD_initial_lr, optimizer, 'SGD')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(SGD_final_lr_fraction, optimizer, 'SGD')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(StepDecay_epochs_per_step, learning_rate_schedule, 'StepDecay')
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(dropout_1, num_layers, 2)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(dropout_2, num_layers, 3)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(dropout_3, num_layers, 4)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(l2_reg_1, num_layers, 2)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(l2_reg_2, num_layers, 3)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(l2_reg_3, num_layers, 4)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(num_units_1, num_layers, 2)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(num_units_2, num_layers, 3)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(num_units_3, num_layers, 4)
    cs.add_condition(cond)
    
    return cs


def objective_function(config, epoch=127, **kwargs):
    # Cast the config to an array such that it can be forwarded to the surrogate
    print(config)
    print(config.get_dictionary())
    
    x = deepcopy(config.get_array())
    x[np.isnan(x)] = -1
    lc = rf.predict(x[None, :])[0]
    c = cost_rf.predict(x[None, :])[0]

    return lc[epoch], {"cost": c, "learning_curve": lc[:epoch].tolist()}


class WorkerWrapper(Worker):
    def compute(self, config, budget, *args, **kwargs):
        cfg = CS.Configuration(cs, values=config)
        loss, info = objective_function(cfg, epoch=int(budget))

        return ({
            'loss': loss,
            'info': {"runtime": info["cost"],
                     "lc": info["learning_curve"]}
        })


if __name__ == '__main__':
    cs = create_config_space()

   

    nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()

    # starting the worker in a separate thread
    w = WorkerWrapper(nameserver=nameserver, ns_port=ns_port)
    w.run(background=True)

    CG = hpbandster.config_generators.RandomSampling(cs)
    
    # instantiating Hyperband with some minimal configuration
    HB = hpbandster.HB_master.HpBandSter(
        config_generator=CG,
        run_id='0',
        eta=2,  # defines downsampling rate
        min_budget=1,  # minimum number of epochs / minimum budget
        max_budget=127,  # maximum number of epochs / maximum budget
        nameserver=nameserver,
        ns_port=ns_port,
        job_queue_sizes=(0, 1),
    )
    # runs one iteration if at least one worker is available
    res = HB.run(10, min_n_workers=1)
    
    # shutdown the worker and the dispatcher
    HB.shutdown(shutdown_workers=True)
    
    # extract incumbent trajectory and all evaluated learning curves
    # traj = res.get_incumbent_trajectory()
    # wall_clock_time = []
    # cum_time = 0
    
    # for c in traj["config_ids"]:
    #     cum_time += res.get_runs_by_id(c)[-1]["info"]["runtime"]
    #     wall_clock_time.append(cum_time)
        
    # lc_hyperband = []
    # for r in res.get_all_runs():
    #     c = r["config_id"]
    #     lc_hyperband.append(res.get_runs_by_id(c)[-1]["info"]["lc"])
        
    # incumbent_performance = traj["losses"]

        

