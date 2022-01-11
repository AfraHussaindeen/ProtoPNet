base_architecture = 'resnet152'
img_size = 224
num_prototypes = 3
num_classes = 10 # Number of unique classes to learn prototypes (Excluding absent classes)
prototype_shape = (num_prototypes * num_classes, 128, 1, 1)
top_k_percentage = 10
prototype_activation_function = "log"
add_on_layers_type = "regular"

# idx_to_class = {0:'bwv_absent', 1:'bwv_present',
#                 2:'dag_absent', 3:'dag_irregular', 4:'dag_regular',
#                 5:'pig_absent', 6: 'pig_irregular', 7:'pig_regular',
#                 8:'pn_absent', 9:'pn_atypical', 10:'pn_typical',
#                 11:'rs_absent', 12:'rs_present',
#                 13:'str_absent', 14:'str_irregular', 15:'str_regular'}
#
# feature_groups = {'bwv':[0,1],
#                   'dag':[2,3,4] ,
#                   'pig':[5,6,7],
#                   'pn':[8,9,10],
#                   'rs': [11,12],
#                   'str': [13,14,15]}

num_labels = 6
num_subclass_labels = {
    0 : 2,
    1 : 3,
    2 : 3,
    3 : 3,
    4 : 2,
    5 : 3
}
class_labels = {0 : 'bwv',
                1 : 'dag',
                2 : 'pig',
                3 : 'pn',
                4 : 'rs',
                5: 'str'}
features = {'bwv':['absent','present'],
              'dag':['absent', 'irregular', 'regular'] ,
              'pig':['absent', 'irregular', 'regular'] ,
              'pn':['absent', 'atypical', 'typical'],
              'rs': ['absent','present'],
              'str': ['absent', 'irregular', 'regular'] }
prototype_label_start_idx = {
    0 : 0,
    1 : 1,
    2 : 3,
    3 : 5,
    4 : 7,
    5 : 8
}
# sub_feature_prototype_class_mapping = {
#     0:{
#         1 : 0 # bwv_present
#     },
#     1 : {
#         1 : 1, #dag_irregular
#         2 : 2  #dag_regular
#     },
#     2 : {
#         1 : 3, #pig_irregular
#         2 : 4  #pig_regular
#     },
#     3 : {
#         1 : 5, #pn_atypical
#         2 : 6  #pn_typical
#     },
#     4:{
#         1 : 7  #rs_present
#     },
#     5:{
#         1 : 8, #str_irregular
#         2 : 9  #str_regular
#     }
# }

experiment_run = '003'

data_csv_path = 'one_hot_dataset.csv'
data_img_path = '../images'
data_path = '../images'
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 1000
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
