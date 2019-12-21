import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

class RNNbase(nn.Module):
    def __init__(self, x_mean, input_size, timestep, hidden_size):
        super().__init__()
        self.x_mean = autograd.Variable(x_mean)
        self.input_size = input_size
        self.timestep = timestep
        # self.batch = 700
        self.hidden_size = hidden_size
        self.h0 = torch.randn(size=(1, self.hidden_size))
        self.output = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=10, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=10, out_features=2),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )
        # 衰减x的
        self.w_gx = nn.Parameter(torch.randn(self.input_size, self.input_size))
        self.h_gx = nn.Parameter(torch.randn(self.input_size, ))

        # 衰减h的
        self.w_gh = nn.Parameter(torch.randn(self.input_size, self.hidden_size))
        self.h_gh = nn.Parameter(torch.randn(self.hidden_size, ))

        print("Timestep is : {}".format(self.timestep))

    def forward(self, *inputs):
        pass

class GRUDT(RNNbase):
    """
    GRU-D
    time weighted
    """
    def __init__(self, **kwargs):
        super(GRUDT, self).__init__(**kwargs)
        self.gru = nn.GRUCell(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              bias=True)
        # 映射 t
        self.w_t = nn.Parameter(torch.randn(1, self.hidden_size))
        self.h_t = nn.Parameter(torch.randn(self.hidden_size, ))


    def forward(self, *inputs):
        x, dt, mask, tp = inputs
        self.h0 = self.h0
        h0 = torch.cat([self.h0 for _ in range(x.shape[0])], dim=0)
        output = []
        xhat = []
        for i in range(self.timestep):
            # 计算衰减率
            gamma_x = torch.exp(
                -torch.max(torch.tensor(0.).to(torch.float32), torch.matmul(dt[:, i, :], self.w_gx) + self.h_gx))
            gamma_h = torch.exp(
                -torch.max(torch.tensor(0.).to(torch.float32), torch.matmul(dt[:, i, :], self.w_gh) + self.h_gh))
            # print("The shape of gamma_h is : {}".format(gamma_h.shape))
            # print("The shape of h0 is : {}".format(h0.shape))
            # 进行衰减
            h0 = gamma_h * h0
            # 进行填补
            if i == 0:
                xs = x[:, i, :] * mask[:, i, :] + (1 - mask[:, i, :]) * self.x_mean
            else:
                xs = x[:, i, :] * mask[:, i, :] + (1 - mask[:, i, :]) * \
                     ((gamma_x * x[:, i - 1, :]) + (1 - gamma_x) * self.x_mean)
            # 映射t
            gt = torch.sigmoid(torch.matmul(tp[:, i, :], self.w_t) + self.h_t)
            # 计算h1
            h1 = self.gru(xs, h0)
            h1 = gt * h1
            # 计算输出
            out = self.output(h1)
            output.append(out)
            # 更新h0
            h0 = h1
            xhat.append(xs)

        # self.h0 = h1[0].view(1, self.hidden_size)
        # print("h0[0] value: {}".format(h0[0]))
        # print("h0[1] value: {}".format(h0[1]))
        # print(h0[0].requires_grad)
        return output, xhat

class GRUD(RNNbase):
    """
    实现gru-d网络
    实现变长处理
    把时间通过mlp映射成为权重
    """
    def __init__(self,**kwargs):
        super(GRUD, self).__init__(**kwargs)
        self.gru = nn.GRUCell(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              bias=True)
    # @torchsnooper.snoop()
    def forward(self, *inputs):
        x, dt, mask, tp = inputs

        self.h0 = self.h0
        h0 = torch.cat([self.h0 for _ in range(x.shape[0])], dim=0)
        output = []
        xhat = []
        for i in range(self.timestep):
            # 计算衰减率
            gamma_x = torch.exp(
                -torch.max(torch.tensor(0.).to(torch.float32), torch.matmul(dt[:, i, :], self.w_gx) + self.h_gx))
            gamma_h = torch.exp(
                -torch.max(torch.tensor(0.).to(torch.float32), torch.matmul(dt[:, i, :], self.w_gh) + self.h_gh))
            # print("The shape of gamma_h is : {}".format(gamma_h.shape))
            # print("The shape of h0 is : {}".format(h0.shape))
            # 进行衰减
            h0 = gamma_h * h0
            # 进行填补
            if i == 0:
                xs = x[:, i, :] * mask[:, i, :] + (1 - mask[:, i, :]) * self.x_mean
            else:
                xs = x[:, i, :] * mask[:, i, :] + (1 - mask[:, i, :]) * \
                     ((gamma_x * x[:, i - 1, :]) + (1 - gamma_x) * self.x_mean)
            # 计算h1
            h1 = self.gru(xs, h0)
            # 计算输出
            out = self.output(h1)
            output.append(out)
            # 更新h0
            h0 = h1
            xhat.append(xs)
        return output, xhat

class LSTMDT(RNNbase):
    """
    实现gru-d网络
    实现变长处理
    把时间通过mlp映射成为权重
    """
    def __init__(self, **kwargs):
        super(LSTMDT, self).__init__(**kwargs)
        self.c0 = torch.randn(size=(1, self.hidden_size))
        self.lstm = nn.LSTMCell(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                bias=True)
        # 映射 t
        self.w_t = nn.Parameter(torch.randn(1, self.hidden_size))
        self.h_t = nn.Parameter(torch.randn(self.hidden_size, ))

    # @torchsnooper.snoop()
    def forward(self, *inputs):
        x, dt, mask, tp = inputs
        # self.h0 = nn.Parameter(self.h0)
        self.h0 = self.h0
        self.c0 = self.c0
        h0 = torch.cat([self.h0 for _ in range(x.shape[0])], dim=0)
        c0 = torch.cat([self.c0 for _ in range(x.shape[0])], dim=0)
        output = []
        xhat = []
        for i in range(self.timestep):
            # 计算衰减率
            gamma_x = torch.exp(
                -torch.max(torch.tensor(0.).to(torch.float32), torch.matmul(dt[:, i, :], self.w_gx) + self.h_gx))
            gamma_h = torch.exp(
                -torch.max(torch.tensor(0.).to(torch.float32), torch.matmul(dt[:, i, :], self.w_gh) + self.h_gh))
            # print("The shape of gamma_h is : {}".format(gamma_h.shape))
            # print("The shape of h0 is : {}".format(h0.shape))
            # 进行衰减
            h0 = gamma_h * h0
            c0 = gamma_h * c0
            # 进行填补
            if i == 0:
                xs = x[:, i, :] * mask[:, i, :] + (1 - mask[:, i, :]) * self.x_mean
            else:
                xs = x[:, i, :] * mask[:, i, :] + (1 - mask[:, i, :]) * \
                     ((gamma_x * x[:, i - 1, :]) + (1 - gamma_x) * self.x_mean)
            # 映射t
            gt = torch.sigmoid(torch.matmul(tp[:, i, :], self.w_t) + self.h_t)
            # 计算h1
            h1, c1 = self.lstm(xs, (h0, c0))
            h1 = gt * h1
            c1 = gt * c1
            # 计算输出
            out = self.output(h1)
            output.append(out)
            # 更新h0
            h0 = h1
            c0 = c1
            xhat.append(xs)

        # self.h0 = h1[0].view(1, self.hidden_size)
        # print("h0[0] value: {}".format(h0[0]))
        # print("h0[1] value: {}".format(h0[1]))
        # print(h0[0].requires_grad)
        return output, xhat

class LSTMD(RNNbase):
    """
    实现gru-d网络
    实现变长处理
    把时间通过mlp映射成为权重
    """
    def __init__(self, **kwargs):
        super(LSTMD, self).__init__(**kwargs)
        self.c0 = torch.randn(size=(1, self.hidden_size))
        self.lstm = nn.LSTMCell(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                bias=True)

        self.output = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=10, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=10, out_features=2),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )

    # @torchsnooper.snoop()
    def forward(self, *inputs):
        x, dt, mask, tp = inputs
        # self.h0 = nn.Parameter(self.h0)
        self.h0 = self.h0
        self.c0 = self.c0
        h0 = torch.cat([self.h0 for _ in range(x.shape[0])], dim=0)
        c0 = torch.cat([self.c0 for _ in range(x.shape[0])], dim=0)
        output = []
        xhat = []
        for i in range(self.timestep):
            # 计算衰减率
            gamma_x = torch.exp(
                -torch.max(torch.tensor(0.).to(torch.float32), torch.matmul(dt[:, i, :], self.w_gx) + self.h_gx))
            gamma_h = torch.exp(
                -torch.max(torch.tensor(0.).to(torch.float32), torch.matmul(dt[:, i, :], self.w_gh) + self.h_gh))
            # print("The shape of gamma_h is : {}".format(gamma_h.shape))
            # print("The shape of h0 is : {}".format(h0.shape))
            # 进行衰减
            h0 = gamma_h * h0
            c0 = gamma_h * c0
            # 进行填补
            if i == 0:
                xs = x[:, i, :] * mask[:, i, :] + (1 - mask[:, i, :]) * self.x_mean
            else:
                xs = x[:, i, :] * mask[:, i, :] + (1 - mask[:, i, :]) * \
                     ((gamma_x * x[:, i - 1, :]) + (1 - gamma_x) * self.x_mean)
            # 计算h1
            h1, c1 = self.lstm(xs, (h0, c0))
            # 计算输出
            out = self.output(h1)
            output.append(out)
            # 更新h0
            h0 = h1
            c0 = c1
            xhat.append(xs)

        # self.h0 = h1[0].view(1, self.hidden_size)
        # print("h0[0] value: {}".format(h0[0]))
        # print("h0[1] value: {}".format(h0[1]))
        # print(h0[0].requires_grad)
        return output, xhat

class PLSTMD(RNNbase):
    """
    实现gru-d网络
    实现变长处理
    把时间通过mlp映射成为权重
    """
    def __init__(self, **kwargs):
        super(PLSTMD, self).__init__(**kwargs)

        self.c0 = torch.randn(size=(1, self.hidden_size))
        self.lstm = nn.LSTMCell(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                bias=True)

        self.tgtype = 1
        # calctimegate 的参数
        if self.tgtype == 1:
            self.s = nn.Parameter(torch.randn(1, self.hidden_size))
            self.tau = torch.randn(1, self.hidden_size)
            self.ratio = nn.Parameter(torch.tensor([[0.8 for _ in range(self.hidden_size)]]))
            self.alpha = 0.01
        else:
            # calctimegate1 的参数
            self.w = nn.Parameter(torch.randn(1, self.hidden_size))
            self.s = nn.Parameter(torch.randn(1, self.hidden_size))
            self.C = nn.Parameter(torch.randn(1, self.hidden_size))
            self.alpha = nn.Parameter(torch.tensor([[0.001 for _ in range(self.hidden_size)]]))

        self.output = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=10, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=10, out_features=2),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )

    def calctimegate(self, time, s, tau, ratio, alpha):

        time = torch.cat([time for _ in range(self.hidden_size)], dim=1)
        fi = torch.fmod((time - s), tau) / tau
        kt = torch.where(fi < ratio / 2, fi * 2 / ratio,
                         torch.where(fi < ratio, 2 - 2 * fi / ratio, alpha * fi))

        # print("The shape of time is : {}".format(time.shape))
        # print("The shape of s is : {}".format(s.shape))
        # print("The shape of tau is : {}".format(tau.shape))
        # print("The shape of ratio is : {}".format(ratio.shape))

        return kt

    def calctimegate1(self, time, w, s, C, alpha):

        time = torch.cat([time for _ in range(self.hidden_size)], dim=1)
        y = torch.sin(w * time + s) + C
        # y1 = torch.where(y > 0, torch.tensor([1.]), alpha)
        y1 = torch.where(y > 0, y, alpha)

        return y1

    # @torchsnooper.snoop()
    def forward(self, *inputs):
        x, dt, mask, tp = inputs
        # self.h0 = nn.Parameter(self.h0)
        self.h0 = self.h0
        self.c0 = self.c0
        h0 = torch.cat([self.h0 for _ in range(x.shape[0])], dim=0)
        c0 = torch.cat([self.c0 for _ in range(x.shape[0])], dim=0)

        if self.tgtype == 1:
            s = torch.cat([self.s for _ in range(x.shape[0])], dim=0).to(torch.float32)
            tau = torch.cat([self.tau for _ in range(x.shape[0])], dim=0).to(torch.float32)
            ratio = torch.cat([self.ratio for _ in range(x.shape[0])], dim=0).to(torch.float32)
        else:
            w = torch.cat([self.w for _ in range(x.shape[0])], dim=0).to(torch.float32)
            s = torch.cat([self.s for _ in range(x.shape[0])], dim=0).to(torch.float32)
            C = torch.cat([self.C for _ in range(x.shape[0])], dim=0).to(torch.float32)
            alpha = torch.cat([self.alpha for _ in range(x.shape[0])], dim=0).to(torch.float32)

        output = []
        xhat = []
        for i in range(self.timestep):
            # 计算衰减率
            gamma_x = torch.exp(
                -torch.max(torch.tensor(0.).to(torch.float32), torch.matmul(dt[:, i, :], self.w_gx) + self.h_gx))
            gamma_h = torch.exp(
                -torch.max(torch.tensor(0.).to(torch.float32), torch.matmul(dt[:, i, :], self.w_gh) + self.h_gh))
            # print("The shape of gamma_h is : {}".format(gamma_h.shape))
            # print("The shape of h0 is : {}".format(h0.shape))
            # 进行衰减
            h0 = gamma_h * h0
            c0 = gamma_h * c0
            # 进行填补
            if i == 0:
                xs = x[:, i, :] * mask[:, i, :] + (1 - mask[:, i, :]) * self.x_mean
            else:
                xs = x[:, i, :] * mask[:, i, :] + (1 - mask[:, i, :]) * \
                     ((gamma_x * x[:, i - 1, :]) + (1 - gamma_x) * self.x_mean)
            # 映射t
            # gt = torch.sigmoid(torch.matmul(tp[:, i, :], self.w_t) + self.h_t)
            # 计算h1
            h1, c1 = self.lstm(xs, (h0, c0))

            # print("The shape of tp is : {}".format(tp[:, i, :].shape))
            if self.tgtype == 1:
                kt = self.calctimegate(time=tp[:, i, :].view(x.shape[0], 1), s=s, tau=tau, ratio=ratio, \
                                       alpha=self.alpha)
            else:
                kt = self.calctimegate1(time=tp[:, i, :].view(x.shape[0], 1), w=w, s=s, C=C, \
                                        alpha=alpha)

            h1 = kt * h1 + (1 - kt) * h0
            c1 = kt * c1 + (1 - kt) * c0

            # h1 = gt * h1
            # c1 = gt * c1
            # 计算输出
            out = self.output(h1)
            output.append(out)
            # 更新h0
            h0 = h1
            c0 = c1
            xhat.append(xs)

        # self.h0 = h1[0].view(1, self.hidden_size)
        # print("h0[0] value: {}".format(h0[0]))
        # print("h0[1] value: {}".format(h0[1]))
        # print(h0[0].requires_grad)
        return output, xhat

class PLSTMDT(RNNbase):
    """
    实现gru-d网络
    实现变长处理
    把时间通过mlp映射成为权重
    """

    def __init__(self, **kwargs):
        super(PLSTMDT, self).__init__(**kwargs)
        self.c0 = torch.randn(size=(1, self.hidden_size))
        self.lstm = nn.LSTMCell(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                bias=True)

        self.tgtype = 1
        # calctimegate 的参数
        if self.tgtype == 1:
            self.s = nn.Parameter(torch.randn(1, self.hidden_size))
            self.tau = torch.randn(1, self.hidden_size)
            self.ratio = nn.Parameter(torch.tensor([[0.8 for _ in range(self.hidden_size)]]))
            self.alpha = 0.01
        else:
            # calctimegate1 的参数
            self.w = nn.Parameter(torch.randn(1, self.hidden_size))
            self.s = nn.Parameter(torch.randn(1, self.hidden_size))
            self.C = nn.Parameter(torch.randn(1, self.hidden_size))
            self.alpha = nn.Parameter(torch.tensor([[0.001 for _ in range(self.hidden_size)]]))

        self.output = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=10, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=10, out_features=2),
            nn.Sigmoid(),
            nn.Softmax(dim=1)
        )


        # 映射 t
        self.w_t = nn.Parameter(torch.randn(1, self.hidden_size))
        self.h_t = nn.Parameter(torch.randn(self.hidden_size, ))

        # 使得在实例中，可以通过属性访问以下变量
        self.__setattr__('w_t', self.w_t)
        self.__setattr__('h_t', self.h_t)

    def calctimegate(self, time, s, tau, ratio, alpha):

        time = torch.cat([time for _ in range(self.hidden_size)], dim=1)
        fi = torch.fmod((time - s), tau) / tau
        kt = torch.where(fi < ratio / 2, fi * 2 / ratio,
                         torch.where(fi < ratio, 2 - 2 * fi / ratio, alpha * fi))

        return kt

    def calctimegate1(self, time, w, s, C, alpha):

        time = torch.cat([time for _ in range(self.hidden_size)], dim=1)
        y = torch.sin(w * time + s) + C
        # y1 = torch.where(y > 0, torch.tensor([1.]), alpha)
        y1 = torch.where(y > 0, y, alpha)

        return y1

    # @torchsnooper.snoop()
    def forward(self, *inputs):
        x, dt, mask, tp = inputs
        # self.h0 = nn.Parameter(self.h0)
        self.h0 = self.h0
        self.c0 = self.c0
        h0 = torch.cat([self.h0 for _ in range(x.shape[0])], dim=0)
        c0 = torch.cat([self.c0 for _ in range(x.shape[0])], dim=0)

        if self.tgtype == 1:
            s = torch.cat([self.s for _ in range(x.shape[0])], dim=0).to(torch.float32)
            tau = torch.cat([self.tau for _ in range(x.shape[0])], dim=0).to(torch.float32)
            ratio = torch.cat([self.ratio for _ in range(x.shape[0])], dim=0).to(torch.float32)
        else:
            w = torch.cat([self.w for _ in range(x.shape[0])], dim=0).to(torch.float32)
            s = torch.cat([self.s for _ in range(x.shape[0])], dim=0).to(torch.float32)
            C = torch.cat([self.C for _ in range(x.shape[0])], dim=0).to(torch.float32)
            alpha = torch.cat([self.alpha for _ in range(x.shape[0])], dim=0).to(torch.float32)

        output = []
        xhat = []
        for i in range(self.timestep):
            # 计算衰减率
            gamma_x = torch.exp(
                -torch.max(torch.tensor(0.).to(torch.float32), torch.matmul(dt[:, i, :], self.w_gx) + self.h_gx))
            gamma_h = torch.exp(
                -torch.max(torch.tensor(0.).to(torch.float32), torch.matmul(dt[:, i, :], self.w_gh) + self.h_gh))
            # print("The shape of gamma_h is : {}".format(gamma_h.shape))
            # print("The shape of h0 is : {}".format(h0.shape))
            # 进行衰减
            h0 = gamma_h * h0
            c0 = gamma_h * c0
            # 进行填补
            if i == 0:
                xs = x[:, i, :] * mask[:, i, :] + (1 - mask[:, i, :]) * self.x_mean
            else:
                xs = x[:, i, :] * mask[:, i, :] + (1 - mask[:, i, :]) * \
                     ((gamma_x * x[:, i - 1, :]) + (1 - gamma_x) * self.x_mean)
                
            # 映射t
            gt = torch.sigmoid(torch.matmul(tp[:, i, :], self.w_t) + self.h_t)
            # 计算h1
            h1, c1 = self.lstm(xs, (h0, c0))

            h1 = gt * h1
            c1 = gt * c1

            # print("The shape of tp is : {}".format(tp[:, i, :].shape))
            if self.tgtype == 1:
                kt = self.calctimegate(time=tp[:, i, :].view(x.shape[0], 1), s=s, tau=tau, ratio=ratio, \
                                       alpha=self.alpha)
            else:
                kt = self.calctimegate1(time=tp[:, i, :].view(x.shape[0], 1), w=w, s=s, C=C, \
                                        alpha=alpha)

            h1 = kt * h1 + (1 - kt) * h0
            c1 = kt * c1 + (1 - kt) * c0

            # h1 = gt * h1
            # c1 = gt * c1
            # 计算输出
            out = self.output(h1)
            output.append(out)
            # 更新h0
            h0 = h1
            c0 = c1
            xhat.append(xs)

        # self.h0 = h1[0].view(1, self.hidden_size)
        # print("h0[0] value: {}".format(h0[0]))
        # print("h0[1] value: {}".format(h0[1]))
        # print(h0[0].requires_grad)
        return output, xhat

MODEL = [PLSTMDT, PLSTMD, LSTMDT, LSTMD, GRUDT, GRUD]