from functions.utils import *

class FeatureOutFCnet(nn.Module):
    def __init__(self, dim_y, DIM, depth, actlayer):
        super(FeatureOutFCnet, self).__init__()

        self.block = [nn.Sequential(nn.Conv2d(dim_y, DIM, 1, 1, padding=0), nn.BatchNorm2d(DIM), actlayer(), )] # 1x1conv to 1x1 representation is same as fc followed by flatten.

        for i in range(depth):
            self.block.append(nn.Sequential(nn.Conv2d(DIM, DIM, 1, 1, padding=0), nn.BatchNorm2d(DIM), actlayer(), ))

        self.block_seq = nn.Sequential(*self.block)

    def forward(self, x):
        x = self.block_seq(x)

        return x

class ConcatDiscriminator(nn.Module):
    def __init__(self, target_model, args):
        super(ConcatDiscriminator, self).__init__()
        DIM = 1024
        actlayer = nn.ReLU

        self.encoder_1 = FeatureOutFCnet(dim_y=16, DIM=DIM, depth=args.before_concat_depth, actlayer=actlayer)
        self.encoder_2 = FeatureOutFCnet(dim_y=160, DIM=DIM, depth=args.before_concat_depth, actlayer=actlayer)
        self.encoder_3 = FeatureOutFCnet(dim_y=320, DIM=DIM, depth=args.before_concat_depth, actlayer=actlayer)
        self.encoder_4 = FeatureOutFCnet(dim_y=640, DIM=DIM, depth=args.before_concat_depth, actlayer=actlayer)

        self.block = [nn.Identity()]

        inter_dim = DIM * 4
        for i in range(args.after_concat_depth):
            inter_dim = inter_dim // 2
            self.block.append(nn.Sequential(nn.Conv2d(inter_dim * 2, inter_dim, 1, 1, padding=0), nn.BatchNorm2d(inter_dim), actlayer(), )) # 1x1conv to 1x1 representation is same as fc followed by flatten.

        self.block_seq = nn.Sequential(*self.block)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inter_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.net1 = torch.nn.Sequential(*list(target_model.children())[:1], torch.nn.AdaptiveAvgPool2d((1, 1)))
        self.net2 = torch.nn.Sequential(*list(target_model.children())[:2], torch.nn.AdaptiveAvgPool2d((1, 1)))
        self.net3 = torch.nn.Sequential(*list(target_model.children())[:3], torch.nn.AdaptiveAvgPool2d((1, 1)))
        self.net4 = torch.nn.Sequential(*list(target_model.children())[:4], torch.nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, global_x, local_x, args):
        x1 = self.net1(global_x)
        x2 = self.net2(global_x)
        x3 = self.net3(local_x)
        x4 = self.net4(local_x)

        if args.layer1_off:
            x1 = x1 * 0
        if args.layer2_off:
            x2 = x2 * 0
        if args.layer3_off:
            x3 = x3 * 0
        if args.layer4_off:
            x4 = x4 * 0

        x1 = self.encoder_1(x1)
        x2 = self.encoder_2(x2)
        x3 = self.encoder_3(x3)
        x4 = self.encoder_4(x4)

        x = torch.cat((torch.cat((x1, x2), 1), torch.cat((x3, x4), 1)), 1)

        x = self.block_seq(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1) #same as x = torch.flatten(x, 1)

        x = self.fc(x)
        return torch.sigmoid(x)
