from pathlib import Path
import os

class BenchLogger(object):
    def __init__(self, name, total_bs, warmup_iter):
        self.name = name
        self.data_time_cpu = AverageMeter()
        self.data_time_gpu = AverageMeter()
        self.batch_time = AverageMeter()
        self.warmup_iter = warmup_iter
        self.total_bs = total_bs
        self.i = 0

    def reset(self):
        self.data_time_cpu.reset()
        self.data_time_gpu.reset()
        self.batch_time.reset()
        self.i = 0

    def iter_callback(self, d, iter_freq=100):
        bt = d['batch_time']
        dt_cpu = d['data_time_cpu']
        dt_gpu = d['data_time_gpu']
        if self.i >= self.warmup_iter:
            self.data_time_cpu.update(dt_cpu)
            self.data_time_gpu.update(dt_gpu)
            self.batch_time.update(bt)
        self.i += 1

        #if self.i % iter_freq == 0:
        #    print("Iter {}: Time: {:.3f}\tData Time: {:.3f}\timg/s (compute): {:.1f}\timg/s (total): {:.1f}".format(
        #      self.i,
        #      self.batch_time.total, self.data_time.total,
        #      self.total_bs / self.batch_time.avg,
        #      self.total_bs / (self.batch_time.avg + self.data_time.avg)))   


    def end_callback(self):
        print("{} summary\tEpoch Time: {:.3f}\tData Time: {:.3f}\timg/s (compute): {:.1f}\timg/s (total): {:.1f}".format(
              self.name,
              self.batch_time.total, self.data_time.total,
              self.total_bs / self.batch_time.avg,
              self.total_bs / (self.batch_time.avg + self.data_time.avg)))
        total_img_s = self.total_bs / (self.batch_time.avg + self.data_time.avg),
        compute_img_s = self.total_bs / self.batch_time.avg,
        prep_img_s = self.total_bs / self.data_time.avg,
        return total_img_s[0], compute_img_s[0], prep_img_s[0], self.batch_time.total, self.data_time.total

    def end_callback(self):
        print("{} summary\tData Time (C): {:.3f}\t Data Time (G): {:.3f}\t Compute Time: {:.3f}\t ".format(
              self.name,
              self.data_time_cpu.total, self.data_time_gpu.total,
              self.batch_time.total))
        #total_img_s = self.total_bs / (self.batch_time.avg + self.data_time.avg),
        #compute_img_s = self.total_bs / self.batch_time.avg,
        #prep_img_s = self.total_bs / self.data_time.avg,
        return self.batch_time.total, self.data_time_cpu.total, self.data_time_gpu.total


class EpochLogger(object):
    def __init__(self, name, total_iterations, args):
        self.name = name
        self.args = args
        self.print_freq = args.print_freq
        self.total_iterations = total_iterations
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.loss = AverageMeter()
        self.time = AverageMeter()
        self.data = AverageMeter()

    def iter_callback(self, epoch, iteration, d):
        self.top1.update(d['top1'], d['size'])
        self.top5.update(d['top5'], d['size'])
        self.loss.update(d['loss'], d['size'])
        self.time.update(d['time'], d['size'])
        self.data.update(d['data'], d['size'])

        if iteration % self.print_freq == 0:
            print('{0}:\t{1} [{2}/{3}]\t'
                  'Time {time.val:.3f} ({time.avg:.3f})\t'
                  #'Total time {time.total:.3f}\t'
                  'Data time {data.val:.3f} ({data.avg:.3f})\t'
                  #'Data time {data.total:.3f}\t'
                  'Speed {4:.3f} ({5:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  self.name, epoch, iteration, self.total_iterations,
                  self.args.batch_size / self.time.val,
                  self.args.batch_size / self.time.avg,
                  time=self.time,
                  data=self.data,
                  loss=self.loss,
                  top1=self.top1,
                  top5=self.top5))

    def epoch_callback(self, epoch):
        print('{0} epoch {1} summary:\t'
              'Time {time.avg:.3f}\t'
              'Data time {data.avg:.3f}\t'
              'Speed {2:.3f}\t'
              'Loss {loss.avg:.4f}\t'
              'Prec@1 {top1.avg:.3f}\t'
              'Prec@5 {top5.avg:.3f}'.format(
              self.name, epoch,
              self.args.batch_size / self.time.avg,
              time=self.time, data=self.data,
              loss=self.loss, top1=self.top1, top5=self.top5))

        self.top1.reset()
        self.top5.reset()
        self.loss.reset()
        self.time.reset()
        self.data.reset()


class PrintLogger(object):
    def __init__(self, train_iterations, val_iterations, args):
        self.train_logger = EpochLogger("Train", train_iterations, args)
        self.val_logger = EpochLogger("Eval", val_iterations, args)

    def train_iter_callback(self, epoch, iteration, d):
        self.train_logger.iter_callback(epoch, iteration, d)

    def train_epoch_callback(self, epoch):
        self.train_logger.epoch_callback(epoch)
        
    def val_iter_callback(self, epoch, iteration, d):
        self.val_logger.iter_callback(epoch, iteration, d)

    def val_epoch_callback(self, epoch):
        self.val_logger.epoch_callback(epoch)
        
    def experiment_timer(self, exp_duration):
        print("Experiment took {} seconds".format(exp_duration))

    def end_callback(self):
        pass


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.total = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.total += val
        self.count += n
        self.avg = self.sum / self.count

class TimingProfiler():
    def __init__(self, log_path, name):
        self.profiler_log = log_path / Path('profiler-' + name + '.csv')
        Path(log_path).mkdir(exist_ok = True)

        with open(self.profiler_log, 'w') as f:
            #f.write("Epoch,Batch size,Synthetic data,Train time,Train (img/s),Data time,Data (img/s),Total time,Total (img/s)\n")
            f.write("Epoch,Batch size,Total time,batch time,data time (CPU),data time (GPU)\n")

    #def write_row(self, epoch, batch_size, synthetic_data, train_time, train_img_s, data_time, data_img_s, total_time, total_img_s):
        #with open(self.profiler_log, 'a') as f:
            #f.write(f"{epoch},{batch_size},{synthetic_data},{round(train_time,2)},{round(train_img_s,2)},{round(data_time,2)},{round(data_img_s,2)},{round(total_time,2)},{round(total_img_s,2)}\n")
    def write_row(self, epoch, batch_size, total_time, batch_time, data_time_cpu, data_time_gpu):
        with open(self.profiler_log, 'a') as f:
            f.write(f"{epoch},{batch_size},{round(total_time,1)},{round(batch_time,1)},{round(data_time_cpu,1)},{round(data_time_gpu,1)}\n")

class GPUProfiler():
    def __init__(self, log_path, name, batch_size, epoch):
        self.profiler_log = Path(log_path) / Path('gpu-profiler-' + name)
        self.profiler_ext = ".csv"
        self.batch_size = batch_size
        self.epoch = epoch
        self.running = False
        Path(log_path).mkdir(exist_ok = True)

    def start(self):
        print("Started GPU stats!")
        log_path = str(self.profiler_log) + f"_bs_{self.batch_size}_epoch_{self.epoch}" + self.profiler_ext
        self.running = True
        os.system(f"nvidia-smi stats -i 0 -d gpuUtil,memUtil -f {log_path} &")

    def stop(self):
        print("Stopped GPU stats!")
        if self.running:
            os.system("pkill -f nvidia-smi")
            self.running = False
