import sys
from keras.applications.inception_v3 import InceptionV3
# from keras.applications.xception import Xception
# from keras.applications.resnet50 import ResNet50
# from keras.applications.mobilenet import MobileNet
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
import ConfigSpace as CS
import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker
import logging
logging.basicConfig(level=logging.DEBUG)

#
#  Testing HpBandSter
#


def get_keras_config_space():
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    "test_parameter",
                    lower=10,
                    upper=20,
                    default_value=15,
                    log=False))
    return cs


def keras_objective(config, epochs, *args, **kwargs):
    """Evaluate success of configuration config."""
    load_model()
    return 0.5, 3, []
    # return loss, runtime, histories


class WorkerWrapper(Worker):    
    def compute(self, config, budget, *args, **kwargs):
        loss, runtime, histories = keras_objective(
            config,
            epochs=int(budget),
            *args, **kwargs)
        return {
            'loss': loss,
            'info': {"runtime": runtime,
                     "histories": histories}
        }


def test_hpbandster(min_budget=1, max_budget=5, job_queue_sizes=(0, 1)):
    nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()
    # starting the worker in a separate thread
    w = WorkerWrapper(nameserver=nameserver, ns_port=ns_port)
    w.run(background=True)
    cs = get_keras_config_space()
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
    
    
    

def load_model():
    pretrained_model = InceptionV3(weights='imagenet', include_top=False)
    # pretrained_model = Xception(weights='imagenet', include_top=False)
    # pretrained_model = ResNet50(weights='imagenet', include_top=False)
    # pretrained_model = MobileNet(weights='imagenet', input_shape=(224, 224,3), include_top=False)
    # pretrained_model = InceptionResNetV2(weights='imagenet', include_top=False)
    return pretrained_model


def simple_test(iterations=20):
    for i in range(iterations):
        print("load model for the {}. time".format(i+1))
        load_model()


if __name__ == "__main__":
    if "--simple" in sys.argv:
        simple_test()
    if "--hpb" in sys.argv:
        test_hpbandster()
