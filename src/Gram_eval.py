from torchvision import transforms
import numpy as np
import torch
from networks import Vgg19
from torch.utils.data import DataLoader
from data import ImageFolder
from progress.bar import Bar
from sklearn.metrics import mean_squared_error
import os


def load_vgg19(model_dir):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg19.weight')):
        vggpth = torch.load(os.path.join('/data/day2night/day2night/UNIT/models/vgg_conv.pth'))
        vgg = Vgg19()
        for (src, dst) in zip(vggpth, vgg.parameters()):
            dst.data[:] = vggpth[src]
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg19.weight'))
    vgg = Vgg19()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg19.weight')))
    return vgg

def get_data_loader_folder_vgg19(input_folder, batch_size, new_size=True,
                           height=1080, width=1080, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.40760392, 0.45795686, 0.48501961),
                                           (1, 1, 1))]
    # transform_list = [transforms.CenterCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize((1080, 1920))] + transform_list if new_size is not None else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform, return_paths=True)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, num_workers=num_workers)
    return loader

def gram_matrix(style_activation):
    # Output activations for each filter convolution
    # Shape is: activations x channels
    style_activation = style_activation.cpu().numpy().squeeze()
    shape = style_activation.shape
    h, w = shape[1], shape[2]
    channel_activations = np.reshape(style_activation, [style_activation.shape[0], -1])

    # Inner Product between all feature activations scaled by total number of activations
    # Shape is: (channels x activations) x (activations x channels) = channels x channels
    gram = channel_activations.dot(channel_activations.T)/(h*w)

    return gram

def get_mse(style_activations, tars):
    assert len(style_activations) == len(tars), 'length not equal'
    length = len(style_activations)

    for i in range(length):
        style_activations_numpy = gram_matrix(style_activations[i])
        tar = tars[i]
        mse_res[i].append(mean_squared_error(style_activations_numpy, tar))
    return 0


def get_folder_content(dir):
    lists = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for i, fname in enumerate(fnames):
            path = os.path.join(root, fname)
            lists.append(path)
        break
    paths = sorted(lists)
    return paths

def read_grams(paths):
    tars = []
    for i in paths:
        tars.append(np.load(i))
    return tars

def accumulate_res(out, idx):
    for i in range(len(out)):
        gram = gram_matrix(out[i])
        if idx == 0:
            outs.append(gram)
        else:
            outs[i] += gram
    return 0

def save_basegram(results, img_num):
    for i in range(len(results)):
        save_npy = results[i]/img_num
        np.save(os.path.join(save_basegram_path, 'base_gram{}'.format(i)), save_npy)
    return 0

def forward(img_path, pth_path, mode_flag):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    loader = get_data_loader_folder_vgg19(img_path, batch_size=1, new_size=True)
    vgg = load_vgg19(pth_path)
    vgg.eval()
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    img_num = len(loader)

    bar = Bar(max=img_num)

    for idx, (img, path) in enumerate(loader):
        score = 0
        path = path[0]
        path = os.path.split(path)[-1]
        img = img.to(device)
        out = vgg(img, style_layers)
        grams_npy_paths = get_folder_content(save_basegram_path)
        tars = read_grams(grams_npy_paths)
        get_mse(out, tars)
        # for i in range(5):
        #     score = score + mse_res[i][0] *weights[i]
        bar.next()
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            idx, len(loader), total=bar.elapsed_td, eta=bar.eta_td)
        print(Bar.suffix)




if __name__ == '__main__':
    # weights = [1e8, 0, 0, 6e2, 0]
    mse_res = [[] for i in range(5)]
    # scores = dict()
    data_path = '/data/day2night/dataset/test/2a6_test'
    model_path = '/data/day2night/day2night/UNIT/models'
    save_basegram_path = '/home/songyang/Music/pic_ori_gram_baseline'
    grams_npy_paths = get_folder_content(save_basegram_path)
    tars = read_grams(grams_npy_paths)
    # mode = 0 # 0 stands for getting base gram matrix, 1 stands for selective mode on
    outs = [] #to save basegram result
    forward(data_path, model_path, mode_flag=0)


        # _ = get_mse(out, tars)
        # for i in range(5):
        #     score = score + mse_res[i][0] *weights[i]
        #
        # scores[path] = score

        # bar.next()
        # Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
        #     idx, len(loader), total=bar.elapsed_td, eta=bar.eta_td)
        # print(Bar.suffix)

    # _ = save_gram(len(loader))
    # np.save('/data/day2night/day2night/UNIT/saved_npy/mse_res_ori/night_test.npy', mse_res)
    # mse_res_mean = [np.mean(i) for i in mse_res]
    np.save('/data/day2night/day2night/UNIT/saved_npy/big_basegram_mseres/2a6.npy', mse_res)
    # np.save('./mse_res/day800_mean.npy', mse_res_mean)
    print()















