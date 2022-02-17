import argparse

def argparser_function():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', '-dataset_path', default='../dataset', help='where is the dataset')
    parser.add_argument('--pth_path', '-pthpath', default='../', help='where is the model files')

    parser.add_argument('--train_or_test', '-tt', default='test', help='option: train, test')
    parser.add_argument('--random_seed', '-seed', type=int, default=42)

    parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')

    parser.add_argument('--main_classifier', '-mc', default='madry', help='option: nat, madry, zhang, lee')
    parser.add_argument('--dataset', '-ds', default='tiny', help='option: cifar10, svhn, cifar100, tiny')

    parser.add_argument('--weight_decay', '-w', type=float, default=2e-4, help='the parameter of l2 restriction for weights')
    parser.add_argument('--lr', '-lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=1, help='max epoch')

    parser.add_argument('--gamma', '-gamma', type=float, default=2, help='adversarial vertex gamma')

    parser.add_argument('--training_eps', '-tr_eps', type=float, default=0.031, help='eps used when training discriminator')
    parser.add_argument('--training_step_size', '-tr_alpha', type=float, default=0.004, help='step size used when training discriminator')
    parser.add_argument('--training_step', '-tr_step', type=int, default=10, help='step used when training discriminator')

    parser.add_argument('--defense_eps', '-def_eps', type=float, default=0.031, help='eps used when purification')
    parser.add_argument('--defense_step_size', '-def_alpha', type=float, default=0.031, help='step size used when purification')
    parser.add_argument('--defense_step', '-def_step', type=int, default=10, help='step used when purification') 

    parser.add_argument('--before_concat_depth', '-bcd', type=int, default=1, help='depth before concat in discriminator')
    parser.add_argument('--after_concat_depth', '-acd', type=int, default=3, help='depth after concat in discriminator')

    parser.add_argument('--layer1_off', '-layer1_off', action='store_true', help='1st conv on/off')
    parser.add_argument('--layer2_off', '-layer2_off', action='store_true', help='5th block on/off')
    parser.add_argument('--layer3_off', '-layer3_off', action='store_true', help='10th block on/off')
    parser.add_argument('--layer4_off', '-layer4_off', action='store_true', help='15th block on/off')

    # added
    parser.add_argument('--n_ex', type=int, default=200)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--attack', type=str)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--ge_iters', type=int)
    parser.add_argument('--ge_eta', type=float)
    parser.add_argument('--use_prior', action='store_true')
    parser.add_argument('--ge_mu', type=float)
    parser.add_argument('--discr_eval_mode', action='store_true')
    parser.add_argument('--init', type=str)
    parser.add_argument('--step_size', type=float)
    parser.add_argument('--zoadamm_lr', type=float, default=.01)
    parser.add_argument('--use_ls', action='store_true')
    parser.add_argument('--eot_iter', type=int, default=0)
    parser.add_argument('--use_custom_discr', action='store_true')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--use_rand_discr', action='store_true')
    
    args = parser.parse_args()

    print('gpu: ', args.gpu)
    print('train or test?: ', args.train_or_test)
    print('dataset: ', args.dataset)
    print('main_classifier: ', args.main_classifier)

    # added
    args.norm = 'Linf'
    args.eps = 12. if args.dataset == 'svhn' else 8.
    
    
    return args