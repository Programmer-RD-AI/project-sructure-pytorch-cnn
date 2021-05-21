from data_loading.data_loader import *
from data_loading.transforming import *
from models.baseline_model import *
from models.final_model import *
from models.test_model import *
from models.testing_models import *
from models.transfer_learning_models import *


def data_loading():
    returned_info = load_all_data()
    return returned_info


def modelling():
    pass


data_loading()
