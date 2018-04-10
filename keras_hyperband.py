# Deep learning lab course final project.
# Kaggle whale classification.

# Hyperparameter optimization on the pretrained Keras model using hyperband.

import logging
logging.basicConfig(level=logging.DEBUG)
import ConfigSpace as CS
import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker
import os
import keras
import keras_model
import keras_tools as tools



def configuration_space_from_raw(hpRaw, hpRawConditions, resolve_multiple='AND'):
    cs = CS.ConfigurationSpace()
    #
    # add hyperparameters
    #
    for hp in hpRaw:
        if hp[4] == "float":
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    hp[0],
                    lower=hp[1][0],
                    upper=hp[1][1],
                    default_value=hp[2],
                    log=hp[3]
                )
            )
        elif hp[4] == "int":
            cs.add_hyperparameter(
                CS.UniformIntegerHyperparameter(
                    hp[0],
                    lower=hp[1][0],
                    upper=hp[1][1],
                    default_value=hp[2],
                    log=hp[3]
                )
            )
        elif (hp[4] == "cat"):
            cs.add_hyperparameter(
                CS.CategoricalHyperparameter(
                    hp[0],
                    hp[1]
                )
            )
        else:
            raise Exception("unknown hp type in hpRawList")

    #
    # add conditions
    #
    covered_conditions = dict()
    for cond in hpRawConditions:
        # check if conditions for that hyperparameter were already processed
        if cond[0] in covered_conditions:
            continue
        covered_conditions[cond[0]] = True
        
        # get all conditions for that hyperparameter
        all_conds_for_hyperparameter = []
        for other_cond in hpRawConditions:
            if other_cond[0] == cond[0]:
                all_conds_for_hyperparameter.append(other_cond)
        
        # create the condition objects
        condition_objects = []
        for cond in all_conds_for_hyperparameter:
            if cond[1] == "eq":
                condition_objects.append(
                    CS.EqualsCondition(
                        cs.get_hyperparameter(cond[0]),
                        cs.get_hyperparameter(cond[2]),
                        cond[3]))
            elif cond[1] == "gtr":
                condition_objects.append(
                    CS.GreaterThanCondition(
                        cs.get_hyperparameter(cond[0]),
                        cs.get_hyperparameter(cond[2]),
                        cond[3]))
            else:
                raise Exception("unknown condition type in hpRawConditions")
        
        # add the conditons to the configuration space
        if len(condition_objects) == 1:
            # simply add the condition
            cs.add_condition(condition_objects[0])
        else:
            # resolve multiple conditions
            if resolve_multiple == 'AND':
                cs.add_condition(
                    CS.AndConjunction(*condition_objects))
            elif resolve_multiple == 'OR':
                cs.add_condition(
                    CS.OrConjunction(*condition_objects))
            else:
                raise Exception("resolve_multiple=", resolve_multiple, ". should be 'AND' or 'OR'")
    
    return cs


# Model the following hyperparameter space:
#   base_model: InceptionV3, ...?
#   num_dense_layers: number of dense classifier layers on top (1 to 4?)
#   num_dense_units_0 ... num_dense_units_3: number of units per dense layer
#   activation: relu, tanh
#   dropout_0 ... dropout_3: dropout probability by layer
#   cnn_unlock_epoch: when to unlock parts of the cnn
#   cnn_num_unlocked: number of pretrained layers to unlock
#   optimizer: Adam, SGD, RMSProp
#   learning_rate
#   batch_size
def get_keras_config_space():
    hpRaw = [
        #<    name              >   <  Range       >      <Default>     <Log>   <Type>
        ["base_model",              ["InceptionV3"],    "InceptionV3",  None,   "cat"],
        ["num_dense_layers",        [1, 4],                 2,          False,  "int"],
        ["num_dense_units_0",       [50, 500],              300,        True,   "int"],
        ["num_dense_units_1",       [50, 500],              200,        True,   "int"],
        ["num_dense_units_2",       [50, 500],              100,        True,   "int"],
        ["num_dense_units_3",       [50, 500],              50,         True,   "int"],
        ["activation",              ["relu", "tanh"],       "relu",     None,   "cat"],
        ["dropout",                 [True, False],          False,      None,   "cat"],
        ["dropout_0",               [0.0, 1.0],             0.5,        False,  "float"],
        ["dropout_1",               [0.0, 1.0],             0.5,        False,  "float"],
        ["dropout_2",               [0.0, 1.0],             0.5,        False,  "float"],
        ["dropout_3",               [0.0, 1.0],             0.5,        False,  "float"],
        ["optimizer",               ["Adam", "SGD", 
                                     "RMSProp"],            "SGD",      None,   "cat"],
        ["learning_rate",           [0.00001, 0.1],         0.001,      True,   "float"],
        ["cnn_unlock_epoch",        [1000, 1500],           1500,       False,  "int"],
        ["unfreeze_percentage",     [0.0, 0.3],             0.1,        False,  "float"],
        ["batch_size",              [16, 64],               32,         True,   "int"],
    ]
    hpRawConditions = [
        #< conditional hp name      >   <cond. Type>    <cond. variable>        <cond. value>
        ["num_dense_units_1",           "gtr",          "num_dense_layers",     1],
        ["num_dense_units_2",           "gtr",          "num_dense_layers",     2],
        ["num_dense_units_3",           "eq",           "num_dense_layers",     4],
        ["dropout_0",                   "eq",           "dropout",              True],
        ["dropout_1",                   "eq",           "dropout",              True],
        ["dropout_2",                   "eq",           "dropout",              True],
        ["dropout_3",                   "eq",           "dropout",              True],
        ["dropout_1",                   "gtr",          "num_dense_layers",     1],
        ["dropout_2",                   "gtr",          "num_dense_layers",     2],
        ["dropout_3",                   "eq",           "num_dense_layers",     4],
    ]
    return configuration_space_from_raw(hpRaw, hpRawConditions, resolve_multiple='AND')


def keras_objective(config, epochs, base_path, *args, **kwargs):
    """Evaluate success of configuration config."""
    loss, runtime, histories = keras_model.train(config, epochs, save_data_path=base_path)
    return loss, runtime, histories


class WorkerWrapper(Worker):
    def __init__(self, save_data_path, objective_function, *args, **kwargs):
        self.save_data_path = save_data_path
        self.objective_function = objective_function
        super().__init__(*args, **kwargs)
    
    def compute(self, config, budget, *args, **kwargs):
        # cfg = CS.Configuration(cs, values=config)
        loss, runtime, histories = self.objective_function(
            config,
            epochs=int(budget),
            base_path=self.save_data_path,
            *args, **kwargs)
        keras.backend.clear_session()  # avoids problems with multithreading
        return {
            'loss': loss,
            'info': {"runtime": runtime,
                     "histories": histories}
        }


def save_plot(x, y, label, ymax=None, path="plot.png"):
    import matplotlib.pyplot as pp
    fig = pp.figure()
    pp.plot(x, y, label=label)
    pp.ylim(ymin=0.0)
    if ymax is not None:
        pp.ylim(ymax=ymax)
    pp.legend()
    pp.savefig(path)


def save_lcs(lcs, label, path="lcs.png"):
    import matplotlib.pyplot as pp
    fig = pp.figure()
    for lc in lcs:
        pp.plot(lc)
    pp.ylim(ymin=0.0, ymax=1.0)
    pp.title(label)
    pp.savefig(path)
    

def optimize(objective=keras_objective, 
             config_space_getter=get_keras_config_space,
             min_budget=1,
             max_budget=127,
             job_queue_sizes=(0, 1),
             base_path="plots",
             run_name=""):

    run_name = tools.get_run_name(prefix="hyperband", additional=run_name)
    path = os.path.join(base_path, run_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    
    nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()
    # starting the worker in a separate thread
    w = WorkerWrapper(save_data_path=path,
                      objective_function=objective,
                      nameserver=nameserver,
                      ns_port=ns_port)
    w.run(background=True)

    cs = config_space_getter()
    configuration_generator = hpbandster.config_generators.RandomSampling(cs)

    # instantiating Hyperband with some minimal configuration
    HB = hpbandster.HB_master.HpBandSter(
        config_generator=configuration_generator,
        run_id='0',
        eta=2,  # defines downsampling rate
        min_budget=min_budget,  # minimum number of epochs / minimum budget
        max_budget=max_budget,  # maximum number of epochs / maximum budget
        nameserver=nameserver,
        ns_port=ns_port,
        job_queue_sizes=job_queue_sizes,
    )
    # runs one iteration if at least one worker is available
    res = HB.run(10, min_n_workers=1)

    # shutdown the worker and the dispatcher
    HB.shutdown(shutdown_workers=True)

    # TODO pickle res
    # TODO save incumbent trajectory as csv

#     # extract incumbent trajectory and all evaluated learning curves
#     traj = res.get_incumbent_trajectory()
#     wall_clock_time = []
#     cum_time = 0

#     for c in traj["config_ids"]:
#         cum_time += res.get_runs_by_id(c)[-1]["info"]["runtime"]
#         wall_clock_time.append(cum_time)

#     lc_hyperband = []
#     for r in res.get_all_runs():
#         c = r["config_id"]
#         lc_hyperband.append(res.get_runs_by_id(c)[-1]["info"]["lc"])

#     incumbent_performance = traj["losses"]

#     # save and plot the wall clock time and the validation of the incumbent
#     # after each iteration here
#     save_plot(wall_clock_time, incumbent_performance,
#               label="validation of incumbent by time",
# #              ymax=0.08,
#               path="incumbent_performance_hb.png")
#     save_lcs(lc_hyperband,
#              label="evaluated learning curves with HyperBand",
#              path="lcs_hyperband.png")
        

if __name__ == "__main__":
    print("optimizing keras_model.")
    optimize(max_budget=5)
