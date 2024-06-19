import torch
import torch.nn as nn
import sys
import r2plus1d
from temporal_shift_module.ops.models import TSN as TSM

def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None

def get_TSM(args,this_weights):
    is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
    if 'RGB' in this_weights:
        modality = 'RGB'
    else:
        modality = 'Flow'
    this_arch = this_weights.split('TSM_')[1].split('_')[2]
    # modality_list.append(modality)
    # num_class, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset, modality)
    print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
    num_class=400
    this_test_segments=8
    net = TSM(num_class, this_test_segments if is_shift else 1, modality,
              base_model=this_arch,
              consensus_type=args.crop_fusion_type,
              img_feature_dim=args.img_feature_dim,
              pretrain=args.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local='_nl' in this_weights,
              )

    if 'tpool' in this_weights:
        from temporal_shift_module.ops.temporal_shift import make_temporal_pool
        make_temporal_pool(net.base_model, this_test_segments)  # since DataParallel

    checkpoint = torch.load(this_weights)
    checkpoint = checkpoint['state_dict']

    # base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
                    }
    for k, v in replace_dict.items():
        if k in base_dict:
            base_dict[v] = base_dict.pop(k)

    net.load_state_dict(base_dict)
    return net


def ResNet50_pretrained():
    import torchvision.models as models
    # import network.resnet as models
    new_model = models.resnet50(pretrained=True)
    new_model.fc = nn.Identity()
    new_model.final_feat_dim = 2048
    # print(new_model)

    return new_model


def Tsm_ResNet50_pretrained():
    # import torchvision.models as models
    import network.resnet as models
    new_model = models.resnet50(pretrained=True,is_tsm=True)
    new_model.fc = nn.Identity()
    new_model.final_feat_dim = 2048
    # print(new_model)

    return new_model

def ResNet50_pretrained_diff():
    import torchvision.models as models
    new_model = models.resnet50(pretrained=True)
    modules = list(new_model.modules())
    new_length = 3
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]

    # modify parameters, assume the first blob contains the convolution kernels
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (new_length *2, ) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous() # 平均后扩展到新增的层

    new_conv = nn.Conv2d(new_length*2, conv_layer.out_channels,
                            conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                            bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

    # replace the first convlution layer
    setattr(container, layer_name, new_conv)
    new_model.fc = nn.Identity()
    new_model.final_feat_dim = 2048
    # print(new_model)
    return new_model


def TsmResNet50_pretrained():
    import argparse
    parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
    # parser.add_argument('dataset', type=str)

    # may contain splits
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--test_segments', type=str, default=25)
    parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
    parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
    parser.add_argument('--full_res', default=False, action="store_true",
                        help='use full resolution 256x256 for test as in Non-local I3D')

    parser.add_argument('--test_crops', type=int, default=1)
    parser.add_argument('--coeff', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')

    # for true test
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--csv_file', type=str, default=None)

    parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='avg')
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--img_feature_dim', type=int, default=256)
    parser.add_argument('--num_set_segments', type=int, default=1,
                        help='TODO: select multiply set of n-frames from a video')
    parser.add_argument('--pretrain', type=str, default='imagenet')

    args = parser.parse_args()

    tsm_model=get_TSM(args,this_weights='/home/denglong/code/FSL-Video/checkpoint/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth')
    tsm_model.fc = nn.Identity()
    tsm_model.final_feat_dim = 2048
    return tsm_model

def ResNet50_pretrained_flow():
    import torchvision.models as models
    new_model = models.resnet50(pretrained=True)
    modules = list(new_model.modules())
    new_length = 2
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]

    # modify parameters, assume the first blob contains the convolution kernels
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (2 * new_length, ) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

    new_conv = nn.Conv2d(2 * new_length, conv_layer.out_channels,
                            conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                            bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

    # replace the first convlution layer
    setattr(container, layer_name, new_conv)
    new_model.fc = nn.Identity()
    new_model.final_feat_dim = 2048
    # print(new_model)
    return new_model

def ResNet50_model():
    import torchvision.models as models
    new_model = models.resnet50(pretrained=False)
    new_model.fc = nn.Identity()
    new_model.final_feat_dim = 2048
    return new_model

class Identity(torch.nn.Module):
    def forward(self, input):
        return input

def ResNet50_tam_pretrained(num_class, num_segments, tam):
    sys.path.append('/home/zzx/workspace/code/temporal-adaptive-module/')
    from ops.models import TSN
    new_model = TSN(num_class, num_segments, 'RGB', 'resnet50', dropout=0.5, tam = tam, partial_bn=False)
    new_model.new_fc = Identity()
    new_model.softmax = Identity()
    new_model.final_feat_dim = 2048
    return new_model

def ResNet50_moco(checkpoint_path = '/home/zzx/workspace/data/pretrained_models/moco_v1_200ep_pretrain.pth.tar'):
    import torchvision.models as models
    new_model = models.resnet50(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = new_model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    new_model.fc = nn.Identity()
    new_model.final_feat_dim = 2048
    return new_model


def sports1m_pretrained(checkpoint_path = '/data/denglong/fsv_pretrained_model/r25d34_sports1m.pth'):
    #import torchvision.models as models
    new_model = r2plus1d.r2plus1d_34(num_classes=5)
    checkpoint = torch.load(checkpoint_path)#map_location='gpu'
    state_dict = checkpoint['state_dict']
    # for k in list(state_dict.keys()):
    #     # retain only encoder_q up to before the embedding layer
    #     if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
    #         # remove prefix
    #         state_dict[k[len("module.encoder_q."):]] = state_dict[k]
    #     # delete renamed or unused k
    #     del state_dict[k]

    msg = new_model.load_state_dict(state_dict, strict=False)
    new_model.final_feat_dim = 2048
    return new_model

