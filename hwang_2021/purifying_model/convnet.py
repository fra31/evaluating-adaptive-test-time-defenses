from functions.utils import *

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)




class FeatureOutConvnet(nn.Module):
    def __init__(self, dim_x, DIM, activation='relu'):
        super(FeatureOutConvnet, self).__init__()

        if activation == 'relu':
            actlayer = nn.ReLU
        elif activation == 'leakyrelu':
            actlayer = nn.LeakyReLU
        else:
            raise print('no act')

        self.conv1 = nn.Conv2d(dim_x, DIM, 3, 2, padding=1)
        self.actfunc1 = actlayer()
        self.conv2 = nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * DIM)
        self.actfunc2 = actlayer()
        self.conv3 = nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * DIM)
        self.actfunc3 = actlayer()
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(4*4*4*DIM, 1)

    def forward(self, x, depth, bn_ON):
        if depth > 0:
            x = self.conv1(x)
            x = self.actfunc1(x)
            # x = self.pool1(x)
            if depth > 1:
                x = self.conv2(x)
                if bn_ON:
                    x = self.bn2(x)
                x = self.actfunc2(x)
                # x = self.pool2(x)
                if depth > 2:
                    x = self.conv3(x)
                    if bn_ON:
                        x = self.bn3(x)
                    x = self.actfunc3(x)
                    # x = self.avgpool(x)
                    # x = x.view(-1, 4*4*4*DIM) #same as x = torch.flatten(x, 1)

                    # x = self.fc(x)
        return x

# class FeatureOutFCnet(nn.Module):
#     def __init__(self, dim_y, DIM, scale_factor, activation='relu'):
#         super(FeatureOutFCnet, self).__init__()
#
#         if activation == 'relu':
#             actlayer = nn.ReLU
#         elif activation == 'leakyrelu':
#             actlayer = nn.LeakyReLU
#         else:
#             raise print('no act')
#
#         self.conv1 = nn.Conv2d(dim_y, DIM * scale_factor, 1, 1, padding=0)
#         self.actfunc1 = actlayer()
#         self.conv2 = nn.Conv2d(DIM * scale_factor, DIM * scale_factor, 1, 1, padding=0)
#         self.bn2 = nn.BatchNorm2d(DIM * scale_factor)
#         self.actfunc2 = actlayer()
#         self.conv3 = nn.Conv2d(DIM * scale_factor, DIM * scale_factor, 1, 1, padding=0)
#         self.bn3 = nn.BatchNorm2d(DIM * scale_factor)
#         self.actfunc3 = actlayer()
#         # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         # self.fc = nn.Linear(4*4*4*DIM, 1)
#
#     def forward(self, x, depth, bn_ON):
#         if depth > 0:
#             x = x.reshape(x.shape[0], x.shape[1], 1, 1)
#             x = self.conv1(x)
#             x = self.actfunc1(x)
#             # x = self.pool1(x)
#
#             if depth > 1:
#                 x = self.conv2(x)
#                 if bn_ON:
#                     x = self.bn2(x)
#                 x = self.actfunc2(x)
#                 # x = self.pool2(x)
#
#                 if depth > 2:
#                     x = self.conv3(x)
#                     if bn_ON:
#                         x = self.bn3(x)
#                     x = self.actfunc3(x)
#                     # x = self.avgpool(x)
#                     # x = x.view(-1, 4*4*4*DIM) #same as x = torch.flatten(x, 1)
#
#                     # x = self.fc(x)
#                     if depth > 3:
#                         raise print('depth error')
#         return x

class FeatureOutFCnet(nn.Module):
    def __init__(self, dim_y, DIM, scale_factor, depth, activation='relu'):
        super(FeatureOutFCnet, self).__init__()

        if activation == 'relu':
            actlayer = nn.ReLU
        elif activation == 'leakyrelu':
            actlayer = nn.LeakyReLU
        else:
            raise print('no act')

        if depth == 1:
            self.block = nn.Sequential(
                nn.Conv2d(dim_y, DIM * scale_factor, 1, 1, padding=0),
                actlayer(),
            )
        elif depth == 2:
            self.block = nn.Sequential(
                nn.Conv2d(dim_y, DIM * scale_factor, 1, 1, padding=0),
                actlayer(),
                nn.Conv2d(DIM * scale_factor, DIM * scale_factor, 1, 1, padding=0),
                nn.BatchNorm2d(DIM * scale_factor),
                actlayer(),
            )
        elif depth == 3:
            self.block = nn.Sequential(
                nn.Conv2d(dim_y, DIM * scale_factor, 1, 1, padding=0),
                actlayer(),
                nn.Conv2d(DIM * scale_factor, DIM * scale_factor, 1, 1, padding=0),
                nn.BatchNorm2d(DIM * scale_factor),
                actlayer(),
                nn.Conv2d(DIM * scale_factor, DIM * scale_factor, 1, 1, padding=0),
                nn.BatchNorm2d(DIM * scale_factor),
                actlayer()
            )
        else:
            raise print('depth error')


    def forward(self, x, depth, bn_ON):
        if depth > 0:
            x = x.reshape(x.shape[0], x.shape[1], 1, 1)

            x = self.block(x)

        return x

class Convnet(nn.Module):
    def __init__(self, input_dim, DIM, feature_depth):
        super(Convnet, self).__init__()
        scale_factor = 1 # 1 when depth 1, 2 whien depth 2, 4 when depth 4
        # concat_scale_factor = 8
        activation = 'relu'

        if activation == 'relu':
            actlayer = nn.ReLU
        elif activation == 'leakyrelu':
            actlayer = nn.LeakyReLU
        else:
            raise print('no act')
        self.encoder_1 = FeatureOutFCnet(dim_y=input_dim[0], DIM=DIM, scale_factor=scale_factor, activation=activation, depth=feature_depth)
        self.encoder_2 = FeatureOutFCnet(dim_y=input_dim[1], DIM=DIM, scale_factor=scale_factor, activation=activation, depth=feature_depth)
        self.encoder_3 = FeatureOutFCnet(dim_y=input_dim[2], DIM=DIM, scale_factor=scale_factor, activation=activation, depth=feature_depth)
        self.encoder_4 = FeatureOutFCnet(dim_y=input_dim[3], DIM=DIM, scale_factor=scale_factor, activation=activation, depth=feature_depth)

        self.conv1 = nn.Conv2d(DIM * 4, DIM * 4, 1, 1, padding=0)
        self.actfunc1 = actlayer()
        self.conv2 = nn.Conv2d(DIM * 4, DIM * 4, 1, 1, padding=0)
        self.bn2 = nn.BatchNorm2d(DIM * 4)
        self.actfunc2 = actlayer()
        self.conv3 = nn.Conv2d(DIM * 4, DIM * 4, 1, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(DIM * 4)
        self.actfunc3 = actlayer()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(DIM * 4, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, x3, x4, device='cuda'): # depth control must consider
        bn_ON = True
        x1 = self.encoder_1(x1, depth=1, bn_ON=bn_ON)
        x2 = self.encoder_2(x2, depth=1, bn_ON=bn_ON)
        x3 = self.encoder_3(x3, depth=1, bn_ON=bn_ON)
        x4 = self.encoder_4(x4, depth=1, bn_ON=bn_ON)
        x = torch.cat((torch.cat((x1, x2), 1), torch.cat((x3, x4), 1)), 1)

        x = self.conv1(x)
        x = self.actfunc1(x)
        x = self.conv2(x)
        if bn_ON:
            x = self.bn2(x)
        x = self.actfunc2(x)
        x = self.conv3(x)
        if bn_ON:
            x = self.bn3(x)
        x = self.actfunc3(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1) #same as x = torch.flatten(x, 1)

        x = self.fc(x)
        return torch.sigmoid(x)


class Discriminator(nn.Module):
    def __init__(self, target_model):
        super(Discriminator, self).__init__()

        self.low = torch.nn.Sequential(*list(target_model.children())[:1], torch.nn.BatchNorm2d(16), torch.nn.ReLU(inplace=True))
        self.high = torch.nn.Sequential(*list(target_model.children())[:4], torch.nn.BatchNorm2d(640), torch.nn.ReLU(inplace=True), torch.nn.AdaptiveAvgPool2d((1, 1)))

        self.global_conv1_1 = nn.Conv2d(640, 2048, 1, 1, padding=0)
        self.global_bn1_1 = nn.BatchNorm2d(2048)
        self.global_relu1_1 = nn.ReLU(inplace=True)
        self.global_conv1_2 = nn.Conv2d(2048, 2048, 1, 1, padding=0)
        self.global_conv2 = nn.Conv2d(640, 2048, 1, 1, padding=0)
        self.global_bn2 = nn.BatchNorm2d(2048)
        self.global_relu2 = nn.ReLU(inplace=True)

        self.local_conv1_1 = nn.Conv2d(16, 2048, 1, 1, padding=0)
        self.local_bn1_1 = nn.BatchNorm2d(2048)
        self.local_relu1_1 = nn.ReLU(inplace=True)
        self.local_conv1_2 = nn.Conv2d(2048, 2048, 1, 1, padding=0)
        self.local_conv2 = nn.Conv2d(16, 2048, 1, 1, padding=0)
        self.local_bn2 = nn.BatchNorm2d(2048)
        self.local_relu2 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, global_x, local_x):
        global_x = self.high(global_x)
        local_x = self.low(local_x)

        global_x_1 = self.global_conv1_1(global_x)
        global_x_1 = self.global_bn1_1(global_x_1)
        global_x_1 = self.global_relu1_1(global_x_1)
        global_x_1 = self.global_conv1_2(global_x_1)
        global_x_2 = self.global_conv2(global_x)
        global_x_2 = self.global_bn2(global_x_2)
        global_x_2 = self.global_relu2(global_x_2)
        global_x = global_x_1 + global_x_2

        local_x_1 = self.local_conv1_1(local_x)
        local_x_1 = self.local_bn1_1(local_x_1)
        local_x_1 = self.local_relu1_1(local_x_1)
        local_x_1 = self.local_conv1_2(local_x_1)
        local_x_2 = self.local_conv2(local_x)
        local_x_2 = self.local_bn2(local_x_2)
        local_x_2 = self.local_relu2(local_x_2)
        local_x = local_x_1 + local_x_2

        local_x = torch.sum(local_x, dim=(2, 3))
        global_x = torch.sum(global_x, dim=(2, 3))

        global_x = global_x * local_x

        global_x = torch.sum(global_x, dim=(1), keepdim=True)

        return torch.sigmoid(global_x)