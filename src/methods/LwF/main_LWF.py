import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from methods.LwF.AlexNet_LwF import AlexNet_LwF
import methods.Finetune.train_SGD as SGD_Training
import utilities.utils as utils
from methods.loss import DiceLoss
from utilities.utils import dice_coefficient, set_lr
def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    print('lr is ' + str(lr))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def Rdistillation_loss(y, teacher_scores, T, scale):
    p_y = F.softmax(y)
    p_y = p_y.pow(1 / T)
    sumpy = p_y.sum(1)
    sumpy = sumpy.view(sumpy.size(0), 1)
    p_y = p_y.div(sumpy.repeat(1, scale))
    p_teacher_scores = F.softmax(teacher_scores)
    p_teacher_scores = p_teacher_scores.pow(1 / T)
    p_t_sum = p_teacher_scores.sum(1)
    p_t_sum = p_t_sum.view(p_t_sum.size(0), 1)
    p_teacher_scores = p_teacher_scores.div(p_t_sum.repeat(1, scale))
    loss = -p_teacher_scores * torch.log(p_y)
    loss = loss.sum(1)

    loss = loss.sum(0) / loss.size(0)
    return loss


def distillation_loss(y, teacher_scores, T, scale):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """

    maxy, xx = y.max(1)
    maxy = maxy.view(y.size(0), 1)
    norm_y = y - maxy.repeat(1, scale)
    ysafe = norm_y / T
    exsafe = torch.exp(ysafe)
    sumex = exsafe.sum(1)
    ######Tscores
    maxT, xx = teacher_scores.max(1)
    maxT = maxT.view(maxT.size(0), 1)
    teacher_scores = teacher_scores - maxT.repeat(1, scale)
    p_teacher_scores = F.softmax(teacher_scores)
    p_teacher_scores = p_teacher_scores.pow(1 / T)
    p_t_sum = p_teacher_scores.sum(1)
    p_t_sum = p_t_sum.view(p_t_sum.size(0), 1)
    p_teacher_scores = p_teacher_scores.div(p_t_sum.repeat(1, scale))

    loss = torch.sum(torch.log(sumex) - torch.sum(p_teacher_scores * ysafe, 1))

    loss = loss / teacher_scores.size(0)
    return loss

# DSL loss for segmentation, HYX
def knowledge_distillation_loss(student_output, teacher_output, temperature):
    # y is the ground truth segmentation map
    # teacher_output and student_output are the output from the teacher and student models
    # alpha is the weight for the distillation loss
    # temperature is the temperature parameter for the distillation

    # Normalize the student's logits
    student_output -= student_output.max(dim=1, keepdim=True)[0]

    # Normalize the teacher's logits
    teacher_output -= teacher_output.max(dim=1, keepdim=True)[0]
    # Calculate the hard target loss (with ground truth labels)
    # hard_loss = F.cross_entropy(student_output, y)

    # Calculate the soft target loss (with teacher's outputs)
    # We use the Kullback-Leibler Divergence loss (KLDivLoss)
    # Note: the teacher's output is detached as we don't want to backpropagate through it
    soft_loss = F.kl_div(F.log_softmax(student_output/temperature, dim=1),
                         F.softmax(teacher_output.detach()/temperature, dim=1),
                         reduction='batchmean')

    # return (1 - alpha) * hard_loss + (alpha * temperature * temperature) * soft_loss
    return soft_loss


def traminate_protocol(since, best_acc):
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


def train_model_lwf(model, original_model, criterion, optimizer, lr, dset_loaders, dset_sizes, use_gpu, num_epochs,
                    exp_dir='./', resume='', temperature=2, previous_model_path='',saving_freq=5, reg_lambda=1,
                    reload_optimizer=True):
    print('dictoinary length' + str(len(dset_loaders)))
    # set orginal model to eval mode
    original_model.eval()

    since = time.time()
    val_beat_counts = 0  # number of time val accuracy not imporved
    best_model = model
    best_acc = 0.0
    mem_snapshotted = False
    preprocessing_time = 0

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        lr = checkpoint['lr']
        print("lr is ", lr)
        val_beat_counts = checkpoint['val_beat_counts']
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    elif os.path.isfile(previous_model_path) and reload_optimizer:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(resume))
        print("load checkpoint from previous task/center model at '{}'".format(previous_model_path))
        checkpoint = torch.load(previous_model_path)
        # model.load_state_dict(checkpoint['state_dict']) # has already been loaded
        lr = checkpoint['lr']
        print("lr is ", lr)
        print('load optimizer')
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(previous_model_path, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(resume))

    print(str(start_epoch))

    for epoch in range(start_epoch, num_epochs + 2):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr, continue_training = set_lr(optimizer, lr, count=val_beat_counts)
                if not continue_training:
                    traminate_protocol(since, best_acc)
                    utils.save_preprocessing_time(exp_dir, preprocessing_time)
                    return model, best_acc
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                start_preprocess_time = time.time()
                # get the inputs
                inputs, labels = data
                # ==========
                if phase == 'train':
                    original_inputs = inputs.clone()

                # wrap them in Variable
                if use_gpu:
                    if phase == 'train':
                        original_inputs = original_inputs.cuda()
                        original_inputs = Variable(original_inputs, requires_grad=False)
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    if phase == 'train':
                        original_inputs = Variable(original_inputs, requires_grad=False)
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                model.zero_grad()
                original_model.zero_grad()
                # forward
                # tasks_outputs and target_logits are lists of outputs for each task in the previous model and current model
                orginal_logits = original_model(original_inputs)
                orginal_logits = torch.cat((1 - orginal_logits, orginal_logits), dim=1) # convert to softmax  HYX
                # Move to same GPU as current model.

                # target_logits = [Variable(item.data, requires_grad=False)
                #                  for item in orginal_logits]
                # print('orginal_logits', orginal_logits.size(), 'target_logits', len(target_logits))
                # del orginal_logits
                # scale = [item.size(-1) for item in target_logits]
                outputs = model(inputs)
                tasks_outputs = torch.cat((1 - outputs, outputs), dim=1) # convert to softmax HYX
                # _, preds = torch.max(tasks_outputs[-1].data, 1)
                preds = torch.round(outputs) # HYX
                task_loss = criterion(outputs, labels)

                # Compute distillation loss.
                dist_loss = 0
                # Apply distillation loss to all old tasks.
                if phase == 'train':
                    # for idx in range(len(target_logits)):
                        # dist_loss += distillation_loss(tasks_outputs[idx], target_logits[idx], temperature, scale[idx])
                    dist_loss += knowledge_distillation_loss(tasks_outputs, orginal_logits, temperature)
                    # backward + optimize only if in training phase
                del orginal_logits
                total_loss = reg_lambda * dist_loss + task_loss
                preprocessing_time += time.time() - start_preprocess_time

                if phase == 'train':
                    total_loss.backward()
                    optimizer.step()

                if not mem_snapshotted:
                    utils.save_cuda_mem_req(exp_dir)
                    mem_snapshotted = True

                # statistics
                running_loss += task_loss.data.item()
                # running_corrects += torch.sum(preds == labels.data).item()
                # print(preds.size(), labels.data.size())

                with torch.no_grad():
                    running_corrects += dice_coefficient(preds, labels).item()

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    del tasks_outputs, labels, inputs, task_loss, preds
                    print('new best val accuracy')
                    best_acc = epoch_acc
                    torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))

                    epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
                    save_checkpoint({
                        'epoch_acc': epoch_acc,
                        'best_acc': best_acc,
                        'epoch': epoch + 1,
                        'lr': lr,
                        'val_beat_counts': val_beat_counts,
                        'arch': 'alexnet',
                        'model': model,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, epoch_file_name)

                    val_beat_counts = 0
                else:
                    val_beat_counts += 1

                if epoch == num_epochs:
                    epoch_file_name = exp_dir + '/' + 'last_epoch' + '.pth.tar'
                    save_checkpoint({
                        'epoch_acc': epoch_acc,
                        'best_acc': best_acc,
                        'epoch': epoch + 1,
                        'lr': lr,
                        'val_beat_counts': val_beat_counts,
                        'arch': 'alexnet',
                        'model': model,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    traminate_protocol(since, best_acc)
    utils.save_preprocessing_time(exp_dir, preprocessing_time)
    return model, best_acc


def fine_tune_LwF_main(dataset_path, previous_task_model_path, init_model_path='', exp_dir='', batch_size=200,
                       num_epochs=100, lr=0.0004, init_freeze=True, pretrained=True, weight_decay=0, last_layer_name=6,
                       saving_freq=5, reg_lambda=1, optimizer=1, reload_optimizer=True):
    print('lr is ' + str(lr))

    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
                    for x in ['train', 'val']}
    # sampler = {}
    # for x in ['train', 'val']:
    #     class_sample_count = [len([idx for idx in range(len(dsets[x])) if dsets[x][idx][1] == t]) for t in range(2)]
    #     weights = 1 / torch.Tensor(class_sample_count)
    #     samples_weight = torch.tensor([weights[t] for _, t in dsets[x]])
    #
    #     sampler[x] = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    #
    # dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], sampler=sampler[x], batch_size=batch_size,
    #                                                shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    #                 for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()
    resume = os.path.join(exp_dir, 'epoch.pth.tar')
    previous_model_path = ''

    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        model_ft = checkpoint['model']
        previous_model = torch.load(previous_task_model_path)
        original_model = copy.deepcopy(previous_model)
        del checkpoint
        del previous_model
    else:
        model_ft = torch.load(previous_task_model_path)

        if 'best_model.pth.tar' in previous_task_model_path:
            previous_model_path = previous_task_model_path.replace('best_model.pth.tar', 'epoch.pth.tar')

        original_model = copy.deepcopy(model_ft)

        #     del init_model
            # do something else
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

    if not hasattr(model_ft, 'reg_params'):
        model_ft.reg_params = {}
    model_ft.reg_params['reg_lambda'] = reg_lambda

    if use_gpu:
        model_ft = model_ft.cuda()
        original_model = original_model.cuda()

    # criterion = nn.CrossEntropyLoss()
    criterion = DiceLoss()
    if optimizer == 0:
        optimizer_ft = optim.SGD(model_ft.parameters(), lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == 1:
        optimizer_ft = optim.Adam(model_ft.parameters(), lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        raise NotImplementedError('Optimizer not implemented. '
                                  'Please set to 0 for SGD or 1 for Adam! Currrent optimizer is ', optimizer)

    model_ft = train_model_lwf(model_ft, original_model, criterion, optimizer_ft, lr, dset_loaders, dset_sizes, use_gpu,
                               num_epochs, exp_dir, resume, temperature=2, previous_model_path=previous_model_path,
                               saving_freq=saving_freq,
                               reg_lambda=reg_lambda,
                               reload_optimizer=reload_optimizer)

    return model_ft


def fine_tune_freeze(dataset_path, model_path, exp_dir, batch_size=100, num_epochs=100, lr=0.0004, optimizer=1):
    print('lr is ' + str(lr))

    dsets = torch.load(dataset_path)
    # dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
    #                                                shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    #                 for x in ['train', 'val']}
    sampler = {}
    for x in ['train', 'val']:
        class_sample_count = [len([idx for idx in range(len(dsets[x])) if dsets[x][idx][1] == t]) for t in range(2)]
        weights = 1 / torch.Tensor(class_sample_count)
        samples_weight = torch.tensor([weights[t] for _, t in dsets[x]])

        sampler[x] = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], sampler=sampler[x], batch_size=batch_size,
                                                   shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    use_gpu = torch.cuda.is_available()
    resume = os.path.join(exp_dir, 'epoch.pth.tar')
    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        model_ft = checkpoint['model']

    model_ft = torch.load(model_path)
    if type(model_ft) is AlexNet_LwF:
        model_ft = model_ft.module
        last_layer_index = str(len(model_ft.classifier._modules) - 1)
        num_ftrs = model_ft.classifier[last_layer_index].in_features
        keep_poping = True
        while keep_poping:
            x = model_ft.classifier._modules.popitem()
            if x[0] == last_layer_index:
                keep_poping = False
    else:
        last_layer_index = str(len(model_ft.classifier._modules) - 1)
        num_ftrs = model_ft.classifier[last_layer_index].in_features

    model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if use_gpu:
        model_ft = model_ft.cuda()

    # criterion = nn.CrossEntropyLoss()
    criterion = DiceLoss()
    if optimizer == 0:
        optimizer_ft = optim.SGD(model_ft.classifier._modules[last_layer_index].parameters(), lr, momentum=0.9)
    elif optimizer == 1:
        optimizer_ft = optim.Adam(model_ft.classifier._modules[last_layer_index].parameters(), lr)
    else:
        raise NotImplementedError('Optimizer not implemented. '
                                  'Please set to 0 for SGD or 1 for Adam! Currrent optimizer is ', optimizer)

    model_ft = SGD_Training.train_model(model_ft, criterion, optimizer_ft, lr, dset_loaders, dset_sizes, use_gpu,
                                        num_epochs, exp_dir, resume)
    return model_ft


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
