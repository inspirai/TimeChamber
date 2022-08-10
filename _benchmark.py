'''
用于测试显卡速度
'''
import os
import torch
import torch.nn as nn
from torch.backends import cudnn
import argparse
import time
import datetime
import platform
from torch.profiler import profile, record_function, ProfilerActivity


def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
                         nn.GroupNorm(16, out_ch, eps=1e-8),
                         act)


def DeConvBnAct(in_ch, out_ch, ker_sz, stride, pad, act=nn.Identity(), group=1, dilation=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, ker_sz, stride, pad, groups=group, bias=False, dilation=dilation),
        nn.GroupNorm(16, out_ch, eps=1e-8),
        act)


class RevSequential(nn.ModuleList):
    '''
    功能大部分与ModuleList重叠
    '''

    def __init__(self, modules=None):
        super().__init__(modules)

    def append(self, module):
        assert hasattr(module, 'invert') and callable(module.invert)
        super().append(module)

    def extend(self, modules):
        for m in modules:
            self.append(m)

    def forward(self, x1, x2):
        y1, y2 = x1, x2
        for m in self:
            y1, y2 = m(y1, y2)
        return y1, y2

    def invert(self, y1, y2):
        x1, x2 = y1, y2
        for m in list(self)[::-1]:
            x1, x2 = m.invert(x1, x2)
        return x1, x2


class RevGroupBlock(RevSequential):
    '''
    当前只支持输入通道等于输出通道，并且不允许下采样
    '''

    def __init__(self, in_ch, out_ch, stride, act, block_type, blocks, **kwargs):
        assert in_ch == out_ch
        assert stride == 1
        mods = []
        for _ in range(blocks):
            mods.append(block_type(in_ch=in_ch, out_ch=out_ch, stride=1, act=act, **kwargs))
        # self.extend(mods)
        super().__init__(mods)


class RevBlockC(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act, **kwargs):
        super().__init__()
        inter_ch = in_ch // 2
        self.conv1 = ConvBnAct(in_ch, inter_ch, ker_sz=5, stride=1, pad=2, act=act)
        self.conv2 = ConvBnAct(inter_ch, inter_ch, ker_sz=5, stride=1, pad=2, act=act, group=inter_ch)
        self.conv3 = ConvBnAct(in_ch, in_ch, ker_sz=1, stride=1, pad=0, act=nn.Identity())

    def func(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y = torch.cat([y1, y2], dim=1)
        y = self.conv3(y)
        return y

    def forward(self, x1, x2):
        y = x1 + self.func(x2)
        return x2, y

    def invert(self, y1, y2):
        x2, y = y1, y2
        x1 = y - self.func(x2)
        return x1, x2


def new_model():
    act = nn.ELU()
    rvb = RevGroupBlock(128, 128, 1, act, RevBlockC, 12).to(device)
    rvb.eval()
    return rvb


if __name__ == '__main__':
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.set_grad_enabled(False)

    parse = argparse.ArgumentParser(description='Used to check pytorch speed benchmark.')
    parse.add_argument('-i', type=int, help='Card id. Which cuda card do you want to test. default: 0', default=0)
    parse.add_argument('-e', type=int, help='Epoch. defaule: 500', default=500)
    parse.add_argument('-bp', type=bool, help='Use backward. defaule: True', default=True)
    parse.add_argument('-bs', type=int, help='Batch size. defaule: 8', default=8)
    parse = parse.parse_args()

    card_id = parse.i
    epoch = parse.e
    use_backward = parse.bp
    batch_size = parse.bs

    # 使用cpu测试理论上是永远不会报错的
    device = 'cpu' if card_id == -1 else f'cuda:{card_id}'
    device = torch.device(device)
    assert epoch > 0
    assert batch_size > 0

    rvb = new_model()

    is_no_num_error = True

    torch.set_grad_enabled(use_backward)

    # start_record = torch.cuda.Event(enable_timing=True)
    # end_record = torch.cuda.Event(enable_timing=True)
    #
    # print('Speed benchmark begin.')
    # start_time = time.perf_counter()
    # start_record.record()
    for e in range(epoch):
        e = e + 1
        with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("step_time"):
                a1 = torch.randn(batch_size, 128, 64, 64, device=device)
                b1, b2 = rvb(a1, a1)
                o_a1, o_a2 = rvb.invert(b1, b2)

                if use_backward:
                    (o_a1.max() + o_a2.max()).backward()

                with torch.no_grad():
                    max_diff_1 = torch.abs(o_a1 - o_a2).max().item()
                    max_diff_2 = torch.abs(a1 - o_a1).max().item()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
        # cur_time = time.perf_counter()
        # cost_time = cur_time-start_time
        # guess_full_cost_time = cost_time / e * epoch
        #
        # line = f'card_id: {card_id} elapsed/total: {e}/{epoch} time: {int(cost_time)}/{int(guess_full_cost_time)} md1: {max_diff_1:.8f} md2: {max_diff_2:.8f}'
        # print(line)

        if max_diff_1 > 1e-3 or max_diff_2 > 1e-3:
            print(f'A large numerical error was found! diff_1: {max_diff_1:.8f} diff_2: {max_diff_2:.8f}')
            is_no_num_error = False

    # end_record.record()
    # torch.cuda.synchronize()
    # end_time = time.perf_counter()
    #
    # cuda_time = start_record.elapsed_time(end_record) / 1000
    # perf_counter_time = end_time - start_time

    # print('Speed benchmark finish.')
    #
    # result = {
    #     'cuda_time': cuda_time,
    #     'perf_counter_time': perf_counter_time,
    #     'no_num_error': is_no_num_error,
    #     'deterministic': cudnn.deterministic,
    #     'benchmark': cudnn.benchmark,
    #     'platform': platform.platform(),
    #     'machine': platform.machine(),
    #     'python_build': platform.python_build(),
    #     'device': 'cpu' if device == torch.device('cpu') else torch.cuda.get_device_name(device),
    #     'test_time': datetime.datetime.now().isoformat(),
    # }
    #
    # print('Result')
    # for k, v in result.items():
    #     print(f'{k}: {v}')
