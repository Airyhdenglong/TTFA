import argparse
import backbone

model_dict = dict(
    ResNet50_pretrained=backbone.ResNet50_pretrained,
    ResNet50_pretrained_diff=backbone.ResNet50_pretrained_diff,
    TsmResNet50_pretrained=backbone.TsmResNet50_pretrained,
    ResNet50_pretrained_flow=backbone.ResNet50_pretrained_flow,
    ResNet50_model=backbone.ResNet50_model,
    ResNet50_tam_pretrained=backbone.ResNet50_tam_pretrained,
    Tsm_ResNet50_pretrained=backbone.Tsm_ResNet50_pretrained,
    ResNet50_moco=backbone.ResNet50_moco,
    sports1m_pretrained=backbone.sports1m_pretrained)


def parse_args(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' % (script))
    parser.add_argument('--dataset', default='somethingotam')
    parser.add_argument('--model', default='ResNet50_pretrained')  # ResNet50_pretrained_1
    parser.add_argument('--flow_model', default='ResNet50_pretrained_flow')  # ResNet50_pretrained_flow
    parser.add_argument('--diff_model', default='ResNet50_pretrained_diff')  # ResNet50_pretrained_flow
    parser.add_argument('--train_n_way', default=5, type=int)
    parser.add_argument('--train_nway', default=10, type=int)
    parser.add_argument('--test_n_way', default=5, type=int)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--train_aug', default=True, type=bool)
    parser.add_argument('--work_dir', default='')
    parser.add_argument('--num_segments', default=8, type=int)
    parser.add_argument('--n_query', default=1, type=int)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--eval_episode', default=1000, type=int)
    parser.add_argument('--test_episode', default=10000, type=int)
    parser.add_argument('--test_model', default=False, type=bool)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint_flow', default=None)
    parser.add_argument('--save_freq', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--stop_epoch', default=-1, type=int)
    parser.add_argument('--temp_set', default=[2, 3], type=list)
    parser.add_argument('--trans_linear_in_dim', default=2048, type=int)
    parser.add_argument('--trans_linear_out_dim', default=1024, type=int)
    parser.add_argument('--trans_dropout', default=0.1, type=float)
    parser.add_argument('--lr', default=1e-5, type=float)

    return parser.parse_args()


def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)