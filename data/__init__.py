from .vca_dataset import VCADataset
from .la2a_dataset import SignalTrainLA2ADataset
from .cl1b_dataset import CL1BDataset

datasets_map = {
    'la2a': {
        'dataset_class': SignalTrainLA2ADataset,
        'train_source': "path_to_la2a_dataset_the_same_in_all_fields",
        'train_targets': "path_to_la2a_dataset_the_same_in_all_fields",
        'val_source': "path_to_la2a_dataset_the_same_in_all_fields",
        'val_targets': "path_to_la2a_dataset_the_same_in_all_fields",
        'test_source': "path_to_la2a_dataset_the_same_in_all_fields",
        'test_targets': "path_to_la2a_dataset_the_same_in_all_fields",
        'nparams': 2        
    },
    'alesis3630': {
        'dataset_class': VCADataset,
        'train_source': "path_to_3630_soruces",
        'train_targets': "path_to_3630_train",
        'val_source': "path_to_3630_soruces",
        'val_targets': "path_to_3630_val",
        'test_source': "path_to_3630_soruces",
        'test_targets': "path_to_3630_test",
        'nparams': 4
    },
    'cl1b': {
        'dataset_class': CL1BDataset,
        'train_source': "path_to_CL1B_train_inputs",
        'train_targets': "path_to_CL1B_train_outputs",
        'val_source': "path_to_CL1B_val_inputs",
        'val_targets': "path_to_CL1B_val_outputs",
        'test_source': "path_to_CL1B_test_inputs",
        'test_targets': "path_to_CL1B_test_outputs",
        'nparams': 4
    }
}