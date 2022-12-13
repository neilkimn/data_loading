import shutil

import torch
from torch import nn
import torch.distributed as dist
from torch.autograd import Variable

from utils.models import ResNet_PT

def should_backup_checkpoint(args):
    def _sbc(epoch):
        return args.gather_checkpoints and (epoch < 10 or epoch % 10 == 0)
    return _sbc

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', backup_filename=None):
    print("SAVING")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    if backup_filename is not None:
        shutil.copyfile(filename, backup_filename)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_prefetched_loader(loader_type):
    if loader_type == "pytorch":
        return prefetched_loader
    if loader_type == "dali":
        return prefetched_loader_dali

def prefetched_loader(loader, device):

    for input, target in loader:
        input = input.to(device)
        target = target.to(device)

        yield input, target

def prefetched_loader_nvidia(loader, device):
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)

    stream = torch.cuda.Stream()
    first = True

    for next_input, next_target in loader:
        with torch.cuda.stream(stream):
            next_input = next_input.cuda()
            next_target = next_target.cuda()
            next_input = next_input.float()
            next_input = next_input.sub_(mean).div_(std)

        if not first:
            yield input, target
        else:
            first = False

        torch.cuda.current_stream().wait_stream(stream)
        input = next_input
        target = next_target

    yield input, target

def prefetched_loader_dali(loader, device):

    for x in loader:
        input, target = x[0]["data"], x[0]["label"]
        target = target.long()
        input = input.to(device)
        target = target.to(device)

        yield input, target


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= torch.distributed.get_world_size()
    return rt

class ModelAndLoss(nn.Module):
    def __init__(self, arch, loss, args):
        super(ModelAndLoss, self).__init__()
        self.arch = arch
        self.num_classes = args.num_classes
        
        print("=> creating model '{}'".format(arch))

        model = ResNet_PT(name = arch, num_classes = self.num_classes)
        criterion = loss()
        criterion = criterion.cuda()

        self.loss = criterion

        model = model.cuda()

        self.model = model

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)
        return loss, output
            
def get_train_step(model_and_loss, optimizer, synthetic_model=False):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)
        if not synthetic_model:
            loss, output = model_and_loss(input_var, target_var)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            reduced_loss = loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()

            return reduced_loss, prec1, prec5
        else:
            _ = model_and_loss(input_var, target_var)

            return 0., 0., 0.

    return _step

def get_val_step(model_and_loss):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            loss, output = model_and_loss(input_var, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step