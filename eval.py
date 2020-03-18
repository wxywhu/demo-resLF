import torch
import os
import numpy as np
from skimage.measure import compare_ssim
from math import log10
from scipy.io import loadmat,savemat
from resLF_model import resLF
from func_input import multi_input_all,uv_list_by_n
import sys
import time
import pandas as pd
from argparse import ArgumentParser, ArgumentTypeError


class Logger:
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Unsupported value encountered.')


def opts_parser():
    usage = "resLF Test"
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '--image_path', type=str, default='/data/test', dest='image_path',
        help='Loading 4D LF images from this path: (default: %(default)s)')
    parser.add_argument(
        '--model', type=str, default='./model/', dest='model_path',
        help='Loading pre-trained model file from this path: (default: %(default)s)')
    parser.add_argument(
        '--scale', type=int, default=2, dest='scale',
        help='Spatial upsampling scale: (default: %(default)s)')
    parser.add_argument(
        '--view_n', type=int, default=7, dest='view_n',
        help='Angular resolution of light field : (default: %(default)s)')
       
    parser.add_argument(
        '--interpolation', type=str, default='bicubic', dest='interpolation',
        help='downsampling interpolation method (`bicubic`): (default: %(default)s)')
    parser.add_argument(
        '--gpu_no', type=int, default=0, dest='gpu_no',
        help='GPU used: (default: %(default)s)')

    return parser


def main(image_path, model_path, scale, view_n=7, is_single=False,
         interpolation='bicubic', gpu_no=0):
    inter_type = ('bicubic', 'blur')
    if interpolation not in inter_type:
        raise ValueError('`{}` interpolation is not supported, Possible values are: bicubic, blur'.format(interpolation))

    # choose GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_no)

    print('=' * 40)
    print('create save directory...')
    save_path=os.path.join(image_path,'Results','ResLFx'+ str(scale))
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('done')
    print('=' * 40)
    print('build network and load model...')
    if is_single:
        model = resLF_model_reading_single(model_path, view_n, scale, interpolation)
    else:
        model = resLF_model_reading_all(model_path, view_n, scale, interpolation)
    print('done')
    print('=' * 40)
    print('predict image...')
    
    xls_list = []
    psnr_list = []
    ssim_list = []
    time_list = []

    filepath =os.path.join(image_path,'x'+ str(scale)) 
    files = os.listdir(filepath)
    for index, image_name in enumerate(files):
        print('-' * 100)
        print('[{}/{}]'.format(index + 1, len(files)), image_name)
        
        lr_y = loadmat(os.path.join(filepath,image_name))['LF_lr']
        hr_y = loadmat(os.path.join(filepath,image_name))['LF_hr']
        
        lr_y=(lr_y/255).clip(16/255, 235/255)
        hr_y=(hr_y/255).clip(16/255, 235/255)
        
        time_item_start = time.time()
        psnr_image, ssim_image, SR_y = predict_y(lr_y, hr_y, view_n, is_single, model, scale)

        time_ = time.time() - time_item_start
        
        #save_mat(save_path, SR_y, image_name[0:-4])


        for i in range(view_n):
            for j in range(view_n):
                print('{:6.4f}/{:6.4f}'.format(psnr_image[i, j], ssim_image[i, j]), end='\t')
            print('')

        print(
                'PSNR Avr: {:.4f}, Max: {:.4f}, Min: {:.4f}, SSIM: Avr: {:.4f}, Max: {:.4f}, Min: {:.4f}, TIME: {:.4f}'
                    .format(np.mean(psnr_image), np.max(psnr_image), np.min(psnr_image),
                            np.mean(ssim_image), np.max(ssim_image), np.min(ssim_image), time_))



        psnr_ = np.mean(psnr_image)
        psnr_list.append(psnr_)
        ssim_ = np.mean(ssim_image)
        ssim_list.append(ssim_)
        time_list.append(time_)

        xls_list.append([image_name, psnr_, ssim_, time_])

    xls_list.append(['average', np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)])
    xls_list = np.array(xls_list)

    result = pd.DataFrame(xls_list, columns=['image', 'psnr', 'ssim', 'time'])
    result.to_csv(image_path + 'resLFx'+str(scale)+'.csv')

    print('-' * 100)
    print('AVR: PSNR: {:.4f}, SSIM: {:.4f}, TIME: {:.4f}'.format(np.mean(psnr_list), np.mean(ssim_list),
                                                                 np.mean(time_list)))
    print('all done')

            
def save_mat(save_path, img, img_name):
    data = np.uint8(np.clip(img.squeeze()*255, 16, 235))        
    save_fn = os.path.join(save_path, img_name+'_SR.mat')
    print(save_fn)
    savemat(save_fn, {'SR_Y': data })

def predict_y(lr_y, hr_y, view_n, is_single, model_dic, scale):
    """
    perdict channel Y
    :param lr_y:
    :param gt_hr_y:
    :param view_n:
    :param model: tuple of model
    :return:
    """
    lr_y, hr_y = torch.from_numpy(lr_y.copy()), torch.from_numpy(hr_y.copy())
    torch.no_grad()
    
    #H = hr_y.shape[2]
    #W = hr_y.shape[3]
    H = lr_y.shape[2]*2
    W = lr_y.shape[3]*2
    psnr_image = np.zeros((view_n, view_n))
    ssim_image = np.zeros((view_n, view_n))
    pr_y = np.zeros((view_n, view_n, H, W), dtype=np.float32)

    uv_dic = uv_list_by_n(view_n)

    for item in range(3, view_n + 1, 2):
        if item == 3:
            model = model_dic['3h']
            u_list = uv_dic['u3h']
            v_list = uv_dic['v3h']
            psnr_image, ssim_image, pr_y = test_all(lr_y, hr_y, view_n, u_list, v_list, model, psnr_image, ssim_image, pr_y, scale)

            model = model_dic['3v']
            u_list = uv_dic['u3v']
            v_list = uv_dic['v3v']
            psnr_image, ssim_image, pr_y = test_all(lr_y, hr_y, view_n, u_list, v_list, model, psnr_image, ssim_image, pr_y, scale)
 
            model = model_dic['3hv']
            u_list = uv_dic['u3hv']
            v_list = uv_dic['v3hv']
            psnr_image, ssim_image, pr_y = test_all(lr_y, hr_y, view_n, u_list, v_list, model, psnr_image, ssim_image, pr_y, scale)
            
        model = model_dic[str(item)]
        u_list = uv_dic['u' + str(item)]
        v_list = uv_dic['v' + str(item)]
        psnr_image, ssim_image, pr_y = test_all(lr_y, hr_y, view_n, u_list, v_list, model, psnr_image, ssim_image, pr_y, scale)

    return psnr_image, ssim_image, pr_y


def resLF_model_reading_all(model_path, view_n, scale, interpolation):
    model_dic = {}
    for item in range(3, view_n + 1, 2):
        task = ['3h', '3v', '3hv']
        if item == 3:
            for i in task:
                model = resLF(n_view=item, scale=scale)
                model.cuda()
                state_dict = torch.load(model_path + '{}_{}_{}.pkl'.format(scale, i, interpolation))
                model.load_state_dict(state_dict)
                model_dic[i] = model

        model = resLF(n_view=item, scale=scale)
        model.cuda()
        state_dict = torch.load(model_path + '{}_{}_{}.pkl'.format(scale, item, interpolation))
        model.load_state_dict(state_dict)
        model_dic[str(item)] = model

    return model_dic


def test_all(test_image, gt_image, view_num_all, u_list, v_list, model, psnr_image, ssim_image, pre_lf, scale):
    for i in range(0, len(u_list), 1):
        u = u_list[i]
        v = v_list[i]

        model.eval()
        train_data_0, train_data_90, train_data_45, train_data_135 = \
            multi_input_all(test_image, view_num_all, u, v, scale)

        train_data_0, train_data_90, train_data_45, train_data_135 = \
            train_data_0.cuda(), train_data_90.cuda(), train_data_45.cuda(), train_data_135.cuda()
            
        # Forward pass: Compute predicted y by passing x to the model
        with torch.no_grad():
            prediction = model(train_data_0, train_data_90, train_data_45, train_data_135)

        output = prediction[0, 0, :, :]
        img_pre = output.cpu().numpy()
        img_pre = np.clip(img_pre, 16/255, 235/255)
        gt_img = gt_image[u,v,:,:].numpy()
        image_h = gt_img.shape[-1]
        image_w = gt_img.shape[-2]

        compare_loss = (img_pre - gt_img) ** 2
        compare_loss = compare_loss.sum() / (image_w * image_h)
        psnr = 10 * log10(1 / compare_loss)
        ssim = compare_ssim(img_pre, gt_img)

        psnr_image[u, v] = psnr
        ssim_image[u, v] = ssim
        pre_lf[u, v, :, :] = img_pre

    return psnr_image, ssim_image, pre_lf


if __name__ == '__main__':
    parser = opts_parser()
    args = parser.parse_args()

    image_path = args.image_path
    model_path = args.model_path
    view_n = args.view_n
    scale = args.scale
    interpolation = args.interpolation
    gpu_no = args.gpu_no

    main(image_path=image_path,
         model_path=model_path,
         scale=scale,
         view_n = view_n,
         interpolation=interpolation,
         gpu_no=gpu_no)
