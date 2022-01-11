import time
import torch

from helpers import list_of_distances
from settings import num_labels, class_labels, features,prototype_label_start_idx
# Evaluation metrics
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score


def _train_or_test(
        model,
        dataloader,
        labels,
        optimizer=None,
        class_specific=True,
        use_l1_mask=True,
        coefs=None,
        log=print,
):
    """
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    """
    is_train = optimizer is not None
    start = time.time()

    pred_list = torch.tensor([])
    target_list = torch.tensor([])

    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0

    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    for i, j in enumerate(dataloader):
        image = j.get('image')
        target = j.get('label') # [0,1,3,1,2,2]
        image = image.cuda()
        target = target.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(image)
            output = output.cuda()
            min_distances = min_distances.cuda()

            # compute loss
            cross_entropy_batch = 0
            cluster_loss_batch = 0
            separation_cost_batch = 0
            avg_separation_cost_batch=0

            for i in range(num_labels) :#Loop through each label (dermoscopic feature)

                # compute cross entropy loss
                cross_entropy_batch += torch.nn.functional.cross_entropy(output[:,i], target[:,i])

                if class_specific:
                    max_dist = (
                            model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3]
                    )

                    prototype_start_idx = prototype_label_start_idx[i]

                    # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                    # for absent classes, it should be all zero tensor
                    temp_target = target[:, i]+(prototype_start_idx-1)
                    temp_target[temp_target<prototype_start_idx] == 0 # For implementation purpose
                    prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,temp_target ]).cuda()
                    prototypes_of_correct_class[target[:, i]==0] = torch.zeros(prototypes_of_correct_class.size()[1])

                    # calculate cluster cost
                    inverted_distances_to_target_prototypes, _ = torch.max(
                        (max_dist - min_distances) * prototypes_of_correct_class, dim=1
                    )
                    cluster_cost = torch.mean(
                        max_dist - inverted_distances_to_target_prototypes
                    )
                    cluster_loss_batch += cluster_cost

                    # calculate separation cost
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    inverted_distances_to_nontarget_prototypes, _ = torch.max(
                        (max_dist - min_distances) * prototypes_of_wrong_class, dim=1
                    )
                    separation_cost = torch.mean(
                        max_dist - inverted_distances_to_nontarget_prototypes
                    )
                    separation_cost_batch += separation_cost

                    # calculate avg separation cost
                    avg_separation_cost = torch.sum(
                        min_distances * prototypes_of_wrong_class, dim=1
                    ) / torch.sum(prototypes_of_wrong_class, dim=1)
                    avg_separation_cost = torch.mean(avg_separation_cost)
                    avg_separation_cost_batch += avg_separation_cost

                    # if use_l1_mask:
                    #     l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    #     l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                    # else:
                    #     l1 = model.module.last_layer.weight.norm(p=1)

                else:
                    min_distance, _ = torch.min(min_distances, dim=1)
                    cluster_cost = torch.mean(min_distance)
                    cluster_loss_batch += cluster_cost
                    # l1 = model.module.last_layer.weight.norm(p=1)


            # if class_specific:
            #     max_dist = (
            #             model.module.prototype_shape[1]
            #             * model.module.prototype_shape[2]
            #             * model.module.prototype_shape[3]
            #     )
            #
            #     # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
            #     prototype_class_identity = model.module.prototype_class_identity.cuda()
            #     t_prototype_class_identity = torch.t(prototype_class_identity)
            #     prototypes_of_correct_class = []
            #     prototypes_of_correct_class_min_distances = torch.tensor([]).cuda()
            #
            #     for i in range(target.size()[0]):
            #
            #         i_label = target[i]
            #         i_min_distances = min_distances[i]
            #
            #         indices = ((i_label == 1).nonzero(as_tuple=True)[0])
            #         i_prototypes_of_correct_class = t_prototype_class_identity[indices]
            #         i_prototypes_of_correct_class = torch.sum(i_prototypes_of_correct_class, axis=0)
            #
            #         prototypes_of_correct_class.append(i_prototypes_of_correct_class)
            #
            #         i_prototypes_of_correct_class_min_distances = torch.tensor([]).cuda()
            #
            #         # enforce the model to have atleast one similar prototype
            #         for index in indices:
            #             inverted_distance = torch.max(
            #                 (max_dist - i_min_distances) * t_prototype_class_identity[index]
            #             )
            #
            #             i_prototypes_of_correct_class_min_distances = torch.cat(
            #                 [i_prototypes_of_correct_class_min_distances,
            #                  torch.unsqueeze((max_dist - inverted_distance), 0)])
            #
            #         prototypes_of_correct_class_min_distances = torch.cat([
            #             prototypes_of_correct_class_min_distances,
            #             i_prototypes_of_correct_class_min_distances])
            #
            #     cluster_cost = torch.mean(prototypes_of_correct_class_min_distances)
            #
            #     prototypes_of_correct_class = torch.stack(prototypes_of_correct_class, dim=0)
            #
            #     # calculate separation cost
            #     prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            #     inverted_distances_to_nontarget_prototypes, _ = torch.max(
            #         (max_dist - min_distances) * prototypes_of_wrong_class, dim=1
            #     )
            #     separation_cost = torch.mean(
            #         max_dist - inverted_distances_to_nontarget_prototypes
            #     )
            #
            #     # calculate avg separation cost
            #     avg_separation_cost = torch.sum(
            #         min_distances * prototypes_of_wrong_class, dim=1
            #     ) / torch.sum(prototypes_of_wrong_class, dim=1)
            #     avg_separation_cost = torch.mean(avg_separation_cost)
            #
            #     if use_l1_mask:
            #         l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
            #         l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
            #     else:
            #         l1 = model.module.last_layer.weight.norm(p=1)
            #
            # else:
            #     min_distance, _ = torch.min(min_distances, dim=1)
            #     cluster_cost = torch.mean(min_distance)
            #     l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            # _, predicted = torch.max(output.data, 1)
            # predicted = predicted.to("cpu")
            # target = target.to("cpu")

            output = output.cpu()
            target = target.cpu()

            pred_list = torch.cat([pred_list, output])  # [[0.23,0.53,0.54] ... x6] | softmax values
            target_list = torch.cat([target_list, target])

            n_batches += 1
            total_cross_entropy += cross_entropy_batch.item()
            total_cluster_cost += cluster_loss_batch.item()
            total_separation_cost += separation_cost_batch.item()
            total_avg_separation_cost += avg_separation_cost_batch.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (
                            coefs["crs_ent"] * cross_entropy_batch
                            + coefs["clst"] * cluster_loss_batch
                            + coefs["sep"] * separation_cost_batch
                            # + coefs["l1"] * l1
                    )
                else:
                    loss = (
                            cross_entropy_batch
                            + 0.8 * cluster_loss_batch
                            - 0.08 * separation_cost_batch
                            # + 1e-4 * l1
                    )
            else:
                if coefs is not None:
                    loss = (
                            coefs["crs_ent"] * cross_entropy_batch
                            + coefs["clst"] * cluster_loss_batch
                            # + coefs["l1"] * l1
                    )
                else:
                    loss = cross_entropy_batch + 0.8 * cluster_loss_batch \
                           # + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del image
        del target
        del output
        del min_distances

    end = time.time()

    class_metrics = get_featurewise_predictions(pred_list, target_list)

    del pred_list
    del target_list

    log("\ttime: \t{0}".format(end - start))
    log("\tcross ent: \t{0}".format(total_cross_entropy / n_batches))
    log("\tcluster: \t{0}".format(total_cluster_cost / n_batches))
    if class_specific:
        log("\tseparation:\t{0}".format(total_separation_cost / n_batches))
        log("\tavg separation:\t{0}".format(total_avg_separation_cost / n_batches))
    log("\tperformance: \t\t{}%".format(class_metrics))
    log("\tl1: \t\t{0}".format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log("\tp dist pair: \t{0}".format(p_avg_pair_dist.item()))

    return class_metrics["pn"]["accuracy"]


def train(
        model, dataloader, labels, optimizer, class_specific=False, coefs=None, log=print
):
    assert optimizer is not None

    log("\ttrain")
    model.train()
    return _train_or_test(
        model=model,
        dataloader=dataloader,
        labels=labels,
        optimizer=optimizer,
        class_specific=class_specific,
        coefs=coefs,
        log=log,
    )


def test(model, dataloader, labels, class_specific=False, log=print):
    log("\ttest")
    model.eval()
    return _train_or_test(
        model=model,
        dataloader=dataloader,
        labels=labels,
        optimizer=None,
        class_specific=class_specific,
        log=log,
    )


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\tlast layer")


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\twarm")


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\tjoint")


def get_performance(feature_output, feature_target, feature_name):
    metric = {}

    outputs_final_label = torch.max(feature_output.data, 1)

    accuracy = accuracy_score(outputs_final_label, feature_target)
    precision = list(precision_score(outputs_final_label, feature_target, average=None))
    recall = list(recall_score(outputs_final_label, feature_target, average=None))
    f1 = list(f1_score(outputs_final_label, feature_target, average=None))
    roc_auc = roc_auc_score(feature_target,feature_output, multi_class='ovr')

    for i, sub_category in enumerate(features[feature_name]):
        metric[sub_category] = {
            "precision": round(precision[i], 3),
            "recall": round(recall[i], 3),
            "f1-score": round(f1[i], 3),
            "roc_auc": round(roc_auc[i], 3)
        }

    metric["accuracy"] = accuracy


    return metric


def get_featurewise_predictions(outputs, targets):
    metric = {}

    for i in range(len(num_labels)):
        feature_name = class_labels[i]

        feature_output = outputs[:,i]
        feature_target = targets[:,i]

        metric[feature_name] = get_performance(feature_output, feature_target, feature_name)

    return metric

# def get_performance(predictions, targets, labels):
#     print("predictions : ", predictions)
#     label_keys = list(labels.values())
#     label_keys.sort()
#     class_metric = {}
#
#     accuracy = accuracy_score(targets, predictions)
#     precision = list(
#         precision_score(targets, predictions, labels=label_keys, average=None)
#     )
#     recall = list(recall_score(targets, predictions, labels=label_keys, average=None))
#     f1 = list(f1_score(targets, predictions, labels=label_keys, average=None))
#
#     for label in labels.keys():
#         class_metric[label] = {
#             "precision": round(precision[int(labels[label])], 3),
#             "recall": round(recall[int(labels[label])], 3),
#             "f1-score": round(f1[int(labels[label])], 3),
#         }
#     class_metric["accuracy"] = accuracy
#
#     return class_metric


# def get_featurewise_predictions(outputs, targets):
#     metric = {}
#     for feature_name in list(feature_groups.keys()):
#
#         feature_metric = {}
#         feature_sub_category =[]
#
#         for i in feature_groups[feature_name]:
#             feature_sub_category.append(idx_to_class[i])
#
#         feature_output = torch.max(outputs[:, feature_groups[feature_name][0]:feature_groups[feature_name][-1]+1], dim=1).indices
#         feature_target = torch.max(targets[:, feature_groups[feature_name][0]:feature_groups[feature_name][-1]+1], dim=1).indices
#
#         feature_accuracy = accuracy_score(feature_target, feature_output)
#
#         feature_precision = list(precision_score(feature_target, feature_output,
#                                                  labels=list(range(len(feature_sub_category))),
#                                                  average=None))
#
#         feature_recall = list(recall_score(feature_target, feature_output,
#                                                  labels=list(range(len(feature_sub_category))),
#                                                  average=None))
#
#         feature_metric['accuracy'] = feature_accuracy
#
#         for i in range(len(feature_sub_category)):
#             feature_metric[feature_sub_category[i]] = {
#                 'precision': feature_precision[i],
#                 'recall': feature_recall[i],
#             }
#
#         metric[feature_name] = feature_metric
#
#     return metric
