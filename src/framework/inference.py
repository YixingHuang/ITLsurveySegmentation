import os.path
import pdb
import torch
import numpy as np

from torch.autograd import Variable
from utilities.utils import dice_coefficient
from PIL import Image
def test_model(method, model, dataset_path, target_task_head_idx, target_head=None, batch_size=200, subset='test',
               per_class_stats=True, final_layer_idx=None, task_idx=None):
    """
    :param target_task_head_idx: for EBLL,LWF which have all heads in model itself
    :param target_head: Actual head in list, so idx should be 0
    """
    if target_head is not None:
        if not isinstance(target_head, list):
            target_head = [target_head]
        assert target_task_head_idx == 0, "Only EBLL, LWF have heads in model itself, here head idx indicates target_headlist idx"

    if hasattr(model, 'classifier'):
        final_layer_idx = str(len(model.classifier._modules) - 1)
    model.eval()
    model = model.cuda()

    # Init dataset
    dsets = torch.load(dataset_path)

    try:
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size, shuffle=False, num_workers=4, drop_last=False)
                        for x in ['train', 'val', 'test']}
    except:
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size, shuffle=False, num_workers=4, drop_last=False)
                        for x in ['train', 'val']}
        print('no test set has been found')
        subset = 'val'
    # dset_classes = dsets['train'].classes

    # Pass args
    holder = type("Holder", (object,), {})()
    holder.task_imgfolders = dsets
    holder.batch_size = batch_size
    holder.model = model
    holder.heads = target_head
    holder.current_head_idx = target_task_head_idx
    holder.final_layer_idx = final_layer_idx
    holder.task_idx = task_idx

    # Init stat counters
    # class_correct = list(0. for i in range(len(dset_classes)))
    # class_total = list(0. for i in range(len(dset_classes)))
    batch_count = 0
    # Iterate data
    running_corrects = 0

    for data in dset_loaders[subset]:
        batch_count += 1
        images, labels, names = data
        images = images.cuda()
        # images = images.squeeze() #cause problems when certain batchs have only one image
        labels = labels.cuda()

        # GET OUTPUT
        outputs = method.get_output(images, holder)
        preds = torch.round(outputs)
        running_corrects += dice_coefficient(preds, labels).item()
        print(torch.sum(images), torch.sum(preds), torch.sum(labels))

        saveBatch2Pngs(images.data.cpu(), 'C:/MachineLearning/CLsurveyMCunet/src/images/', task_idx, batch_count, suffix='_input', original_names=names)
        saveBatch2Pngs(preds.data.cpu(), 'C:/MachineLearning/CLsurveyMCunet/src/images/', task_idx, batch_count, suffix='_pred', original_names=names)
        saveBatch2Pngs(labels.data.cpu(), 'C:/MachineLearning/CLsurveyMCunet/src/images/', task_idx, batch_count, suffix='_gt', original_names=names)
        # _, target_head_pred = torch.max(outputs.data, 1)
        # c = (target_head_pred == labels).squeeze()
        #
        # # Class specific stats
        # for i in range(len(target_head_pred)):
        #     label = labels[i].item()
        #     class_total[label] += 1
        #     class_correct[label] += c.item() if len(c.shape) == 0 else c[i].item()

        del images
        del labels
        del outputs
        del data
        del preds

    # Final postprocessing
    # Per class ACC
    # if per_class_stats:
    #     print("For all correct-head classified: NO Results for segmentation tasks")
        # for i in range(len(dset_classes)):
        #     print('Accuracy of %5s : %2d %%' % (
        #         dset_classes[i], 100 * class_correct[i] / class_total[i]))

    # OVERALL ACC
    # accuracy = np.sum(class_correct) * 100 / (np.sum(class_total))
    accuracy = running_corrects / len(dset_loaders[subset].dataset)
    print('Overall Accuracy: ' + str(accuracy))

    return accuracy

def saveBatch2Pngs(batch, save_path,  task_idx, batch_idx, suffix='', original_names=None):
    batch = batch * 255
    for i in range(batch.shape[0]):
        img = Image.fromarray(batch[i, 0, :, :].numpy().astype(np.uint8))
        # img.save(save_path + str(task_idx) + '_' + str(batch_idx) + '_' + str(i) + suffix + '.png')
        img.save(save_path + str(task_idx + 1) + '_' + os.path.basename(original_names[i])[0:-5] + '_' + suffix + '.png')


def test_task_joint_model(model_path, dataset_path, task_idx, task_lengths, batch_size=200, subset='test',
                          print_per_class_acc=True, debug=False, tasks_idxes=None, init_freeze=True):
    """
    Test the performance of a given task in a model that is trained jointly on a set of tasks.
    Shared output layer, but masks out other task outputs.

    :param model_path:
    :param dataset_path:
    :param task_idx: the tested task ordered idx in the task lengths
    :param task_lengths: number of classes in each task
    :param batch_size:
    :param subset:
    :param print_per_class_acc:
    :param debug:
    :param tasks_idxes: array of lists, with each list a set of integers that correspond to the FC outputs for this task_idx
    :return:
    """
    print("==> TESTING TASK {}".format(task_idx + 1))

    model = torch.load(model_path)
    model.eval()
    model = model.cuda()
    dsets = torch.load(dataset_path)
    try:
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size, shuffle=True, num_workers=4)
                        for x in ['train', 'val', 'test']}
    except:
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size, shuffle=True, num_workers=4)
                        for x in ['train', 'val']}
        print('no test set has been found')
        subset = 'val'
    dset_classes = dsets['train'].classes
    class_correct = list(0. for i in range(len(dset_classes)))
    class_total = list(0. for i in range(len(dset_classes)))
    running_corrects = 0
    if init_freeze:
        this_task_class_mask = tasks_idxes
    else:
        if tasks_idxes is None:
            this_task_class_mask = torch.tensor(list(range(task_lengths[task_idx]))) + sum(task_lengths[0:task_idx])
        else:
            this_task_class_mask = tasks_idxes[task_idx]
            assert isinstance(this_task_class_mask, list)

    if debug:
        print("TESTING PARAMS:")
        print("this_task_class_mask={}".format(this_task_class_mask))

    for data in dset_loaders[subset]:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(Variable(images))

        preds = torch.round(outputs)
        running_corrects += dice_coefficient(preds, labels).item()


        del images
        del labels
        del outputs
        del data

    if print_per_class_acc:
        for i in range(len(dset_classes)):
            print('Accuracy of %5s : %2d %%' % (
                dset_classes[i], 100 * class_correct[i] / class_total[i]))
    if debug:
        pdb.set_trace()

    accuracy = running_corrects / len(dset_loaders[subset].dataset)

    print('Accuracy: ' + str(accuracy) )
    return accuracy
