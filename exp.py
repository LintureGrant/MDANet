import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from model_MD import MDANet
from tqdm import tqdm
from API import *
from utils import *
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from thop import profile
import torch.distributed
from timm.utils.agc import adaptive_clip_grad
from torch import optim

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')

        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = MDANet( shape_in=tuple(args.in_shape),
                            shape_out=tuple(args.out_shape),
                            hid_S=args.hid_S,
                            hid_T=args.hid_T,
                            N_S=args.N_S,
                            kernel_size=tuple(args.kernel_size),
                            N_T=args.N_T,
                            rednet_deep=args.rednet_deep,
                            layer_config=tuple(args.layer_config),
                            groups=args.groups).to(self.device)
        input = torch.randn(16, 10, 1, 64, 64).to("cuda")
        flops, params = profile(self.model, inputs=(input,))
        print('flops:', flops)
        print('params:', params)
        self.model.load_state_dict(torch.load(self.path + '/' + '600.pth')['net'], strict=False)

    def _get_data(self):
        config = self.args.__dict__
        print("*****************************")
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

        #在这里获得句柄


    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)

        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.lr, total_steps=625 * self.args.epochs, final_div_factor=10000.0)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()


    def _save(self, name=''):

        checkpoint = {
            'net': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.checkpoints_path, name + '.pth'))



    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)

        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)
            j=0
            for batch_x, batch_y in train_pbar:
                j=j+1
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)

                loss = self.criterion(pred_y, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                train_pbar.set_description('train loss: {:.4f}, learning rate: {:.6f}'.format(loss.item(), lr))

                self.optimizer.step()
                self.scheduler.step()
            torch.cuda.empty_cache()

            train_loss = np.average(train_loss)

            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader)
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch))
                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}  lr: {3:.7f}\n".format(
                    epoch + 1, train_loss, vali_loss, lr))
                recorder(vali_loss, self.model, self.path)

        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            loss = self.criterion(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        mse, mae, ssim, psnr = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, True)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        self.model.train()
        return total_loss

    def test(self, args):
        self.model.eval()
        count=0
        inputs_lst, trues_lst, preds_lst = [], [], []
        for batch_x, batch_y in self.test_loader:
            print(count)
            pred_y = self.model(batch_x.to(self.device))
            loss = self.criterion(pred_y.to(self.device), batch_y.to(self.device))
            print(loss.item()*64*64)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))


        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mse, mae, ssim, psnr = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, True)
        print_log('mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return mse