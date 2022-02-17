import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from test_models import testing
from Adversarial_attack import fgsm, Random, pgd_unconstrained

def truncated_normal(size):
    values = torch.fmod(torch.randn(size), 2) * 8
    return values

def label_smooth(y, num_classes, weight=0.9):
    # requires y to be one_hot!
    return F.one_hot(y, num_classes=num_classes).type(torch.float).clamp(min=(1. - weight) / (num_classes - 1.), max=weight)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
    
def random_flip_left_right(images):
    images_flipped = torch.flip(images, dims=[3])
    flip = torch.bernoulli(torch.ones(images.shape[0],) * 0.5).type(torch.bool)
    images[flip] = images_flipped[flip]
    images = images.detach()
    return images
        
def training(train_loader, test_loader, model, epochs, start_epoch, learning_rate, lr_schedule, 
             momentum, weight_decay, num_classes, train_method=None, device=None):
    best_accuracy = 0.
    best_prec_history = 0.
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                milestones=lr_schedule, gamma=0.1)
    for epoch in range(start_epoch, epochs):
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        if train_method == 'standard':
            train_standard(train_loader, model, criterion, optimizer, epoch, epochs, device)
        elif train_method == 'label_smooth':
            train_label_smooth(train_loader, model, criterion, optimizer, epoch, epochs, num_classes, device)
        else:
            train_adversarial(train_loader, model, criterion, optimizer, epoch, epochs, device, train_method)
        lr_scheduler.step() 
        
        accuracy = testing(test_loader, model, step_size=0., eps=0., device=device)
        print ('Acc: {:.3f}'.format(accuracy))
        best_accuracy = max(accuracy, best_accuracy)
        if best_accuracy > best_prec_history:
            best_prec_history = best_accuracy
            save_checkpoint(model.state_dict(), filename='Model_trained_by_{}.ckpt'.format(train_method))

def train_standard(train_loader, model, criterion, optimizer, epoch, num_epochs, device):
    train_loss = 0
    train_correct = 0
    train_total = 0
    total_step = len(train_loader)
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        images = random_flip_left_right(images)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        if (i+1) % 100 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], LR: [{}], Loss: {:.4f}, Acc: {:.3f}' 
                               .format(epoch+1, num_epochs, i+1, total_step,
                                       optimizer.param_groups[0]['lr'], train_loss/total_step, 100.*train_correct/train_total))        

def cross_entropy(logits, targets):
    return (-targets * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()
    
def train_label_smooth(train_loader, model, criterion, optimizer, epoch, num_epochs, num_classes, device):
    train_loss = 0
    train_correct = 0
    train_total = 0
    total_step = len(train_loader)
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        images = random_flip_left_right(images)
        labels_ = label_smooth(labels, num_classes)
        outputs = model(images)
        loss = cross_entropy(outputs, labels_)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        if (i+1) % 100 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], LR: [{}], Loss: {:.4f}, Acc: {:.3f}' 
                               .format(epoch+1, num_epochs, i+1, total_step,
                                       optimizer.param_groups[0]['lr'], train_loss/total_step, 100.*train_correct/train_total)) 


def train_adversarial(train_loader, model, criterion, optimizer, epoch, num_epochs, device, train_method):
    train_loss = 0
    train_correct = 0
    train_total = 0
    total_step = len(train_loader)
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        eps_defense = torch.abs(truncated_normal(images.shape[0]))
        num_normal = int(images.shape[0] / 2)
        images, labels = images.to(device), labels.to(device)
        images = random_flip_left_right(images)
        outputs = model(images)
        _, predicted_label = outputs.max(1)
        if train_method == 'fgsm':
            images_adv_ = fgsm(images, predicted_label, eps_defense, criterion, model)
        elif train_method == 'random':
            images_adv = Random(images, predicted_label, eps_defense, criterion, model)
        elif train_method == 'pgd':
            random_number = torch.abs(truncated_normal(1))
            attack_iter = int(min(random_number + 4, 1.25 * random_number))
            images_adv = pgd_unconstrained(model, images, predicted_label, criterion, num_steps=attack_iter, step_size=1)
        images_final = images_adv.detach()
        outputs_worse = model(images_final)
        loss_1 = criterion(outputs_worse[0:num_normal], labels[0:num_normal]) * 2.0 / 1.3
        loss_2 = criterion(outputs_worse[num_normal:], labels[num_normal:]) * 0.6 / 1.3
        loss = (loss_1 + loss_2) / 2.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs_worse.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        if (i+1) % 100 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], LR: [{}], Loss: {:.4f}, Acc: {:.3f}' 
                               .format(epoch+1, num_epochs, i+1, total_step,
                                       optimizer.param_groups[0]['lr'], train_loss/total_step, 100.*train_correct/train_total))