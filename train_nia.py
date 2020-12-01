import argparse
import os
from time import time

import torch


from nia_data import NIADataset
from logger import Logger
from loss import lovasz_softmax, dice_bce_loss
from networks.dinknet import LinkNet34, DinkNet34
from networks.nllinknet_location import NL3_LinkNet, NL4_LinkNet, NL34_LinkNet, Baseline
from networks.nllinknet_pairwise_func import NL_LinkNet_DotProduct, NL_LinkNet_Gaussian, NL_LinkNet_EGaussian
from networks.unet import Unet
from test_nia import test_models
from train_framework import TrainFramework


def train_models(model, name, crop_size=(1024, 1024), init_learning_rate=0.0003, dataset_root='../dataset/Road/train/',
                 load='', BATCHSIZE_PER_CARD=4,total_epoch=500,weight_decay_factor=5.0, num_classes=7):
    if type(crop_size) == tuple:
        crop_size = list(crop_size)

    print(model, name, crop_size, init_learning_rate, dataset_root, load)

    Loss = lovasz_softmax
    # Loss = dice_bce_loss()
    #imagelist = list(filter(lambda x: x.find('sat') != -1, os.listdir(dataset)))
    #trainlist = list(map(lambda x: x[:-8], imagelist))

    solver = TrainFramework(model, Loss, init_learning_rate, num_classes=num_classes)
    if load != '':
        print('Loading...')
        solver.load(load)
    batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

    #dataset = ImageFolder(trainlist, dataset, crop_size)
    dataset = NIADataset(dataset_root, crop_size[0], shuffle=True, rgb=True, n_class=num_classes)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=BATCHSIZE_PER_CARD,
        drop_last=True)

    mylog = Logger('logs/' + name + '.log')
    tic = time()
    no_optim = 0
    train_epoch_best_loss = 100.
    for epoch in range(1, total_epoch + 1):

        data_loader_iter = iter(data_loader)

        train_epoch_loss = 0

        for batch in data_loader_iter:
            img = batch["img"]
            mask = batch["mask"]
            # [8,3,1024,1024], [8,1,1024,1024]
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(data_loader_iter)

        mylog.write('********\n')
        mylog.write('epoch:' + str(epoch) + '    time:' + str(int(time() - tic)) + '\n')
        mylog.write('train_loss:' + str(train_epoch_loss) + '\n')
        mylog.write('SHAPE:' + str(crop_size) + '\n')

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('weights/' + name + '_{}.th'.format(epoch))

        """
        if no_optim > 8 and False:  # 6
            mylog.write('early stop at %d epoch' % epoch)
            break
        if no_optim >= 2:  # 3
            if solver.old_lr < 5e-6 and False:  # 5e-7
                break
            # solver.load('weights/' + name + '.th')
            solver.update_lr(weight_decay_factor, factor=True, mylog=mylog)
            no_optim = 0
            #train_epoch_best_loss = train_epoch_loss
        """
        mylog.flush()

    mylog.write('Finish!')
    mylog.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="set the model name")
    parser.add_argument("--name", help="name of log and weight files")
    parser.add_argument("--crop_size", help="set the crop size", default=[1024, 1024], type=int, nargs='*')
    parser.add_argument("--init_lr", help="set the initial learning rate", default=0.0008, type=float)
    parser.add_argument("--dataset", help="the path of train datasets", default="dataset/train")
    parser.add_argument("--load", help="the path of the weight file for loading", default="")
    parser.add_argument("--total_epoch", help="total number of epochs", type=int, default=500)
    parser.add_argument("--weight_decay_factor", help="wegith decay factor", type=float, default=5.0)
    parser.add_argument("--num_cls", help="the number of classes", default=7, type=int)
    parser.add_argument("--batch_per_gpu", help="the batch size for a GPU", default=8, type=int)
    args = parser.parse_args()

    models = {'NL3_LinkNet': NL3_LinkNet, 'NL4_LinkNet': NL4_LinkNet, 'NL34_LinkNet': NL34_LinkNet,
              'Baseline': Baseline,
              'NL_LinkNet_DotProduct': NL_LinkNet_DotProduct, 'NL_LinkNet_Gaussian': NL_LinkNet_Gaussian,
              'NL_LinkNet_EGaussian': NL_LinkNet_EGaussian,
              'UNet': Unet, 'LinkNet': LinkNet34, 'DLinkNet': DinkNet34}

    model = models[args.model]
    name = args.name
    crop_size = args.crop_size
    init_learning_rate = args.init_lr
    dataset = args.dataset
    load = args.load
    total_epoch = args.total_epoch
    weight_decay_factor = args.weight_decay_factor
    
    train_models(model=model, name=name, crop_size=crop_size, init_learning_rate=init_learning_rate, dataset_root=dataset,
                 load=load, total_epoch=total_epoch, weight_decay_factor=weight_decay_factor, num_classes=args.num_cls, BATCHSIZE_PER_CARD=args.batch_per_gpu)
    # test_models(model=model, n_class=7, name=name, scales=[1.0], source=dataset)

if __name__ == "__main__":
    main()
