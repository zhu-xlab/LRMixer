import os
import time
import argparse
import numpy as np

from Net import Net, RCLoss
import torch
import torch.optim as optim
from dataset import Dataset_syn,Dataset_real

from metric import calc_psnr, calc_sam, calc_ergas, calc_rmse

parser = argparse.ArgumentParser(description='MHFusion')
parser.add_argument('--epochs', type=int, default=10000, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
parser.add_argument('--CUDA', type=str, default='0',help='CUDA')
parser.add_argument('--resume', dest='resume', action='store_true', default=False, help='if resume')
parser.add_argument('--exp_name', type=str, default='pavia4', help='dataset name')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA

torch.backends.cudnn.benchmark = True
if torch.backends.cudnn.benchmark:
    torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.makedirs(args.exp_name, exist_ok=True)
os.makedirs(os.path.join(args.exp_name, 'model'), exist_ok=True)
os.makedirs(os.path.join(args.exp_name, 'gt'), exist_ok=True)
os.makedirs(os.path.join(args.exp_name, 'out'), exist_ok=True)

if 'pavia4' in args.exp_name:
    test_epoch = 1
    file_path = './data/PaviaC/Pavia.mat'
    data_key = 'pavia'
    hc = 102
    mc = 3
    ds = 4
    dataset = Dataset_syn(file_path, data_key, patch_size=64, ratio=ds, crop_ratio=0.8, mc=mc, iters=10000)
if args.exp_name == 'chikusei4':
    test_epoch = 1
    file_path = './data/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat'
    data_key = 'chikusei'
    hc = 128
    mc = 3
    ds = 4
    dataset = Dataset_syn(file_path, data_key, patch_size=64, ratio=ds, crop_ratio=0.8, mc=mc, iters=10000)
if args.exp_name == 'mdas':
    test_epoch = 1
    file_path = './data/mdas/Augsburg_data_4_publication'
    hc = 242
    mc = 4
    ds = 3
    dataset = Dataset_real(file_path, patch_size=48, iters=10000)

net = Net(mc, hc, ds)
net.to(device)
criterion = RCLoss()
criterion.to(device)
optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.epochs//2, gamma=0.1)

def torch2np(tensor):
    tensor = tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    return tensor

if args.resume:
    model_path = os.path.join(args.exp_name, 'model', 'checkpoint.pth.tar')
    assert os.path.isfile(model_path)
    checkpoint = torch.load(model_path)
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Load checkpoint at epoch %d.' % start_epoch)
else:
    start_epoch = 0

def eval(net, dataset, epoch):
    psnr = []
    ssim = []
    gts, lrhsis, hrmsis = dataset.get_test() 

    with torch.no_grad():
        net.eval()
        for i in range(gts.shape[0]):
            gt = gts[i:i+1].float()
            h = lrhsis[i:i+1].float()
            m = hrmsis[i:i+1].float()
            
            gt = gt.to(device)
            h = h.to(device)
            m = m.to(device)
            out = net(m, h)

            out = out.clip(0,1)
            psnr.append(calc_psnr(torch2np(out), torch2np(gt)))
            ssim.append(calc_sam(torch2np(out), torch2np(gt)))
                
    print("epoch:{}, test psnr:{}, sam:{}".format(epoch, np.mean(psnr).round(4), np.mean(ssim).round(4)))

for epoch in range(start_epoch, args.epochs, 1):
    net.train()
    start = time.time()
    gt, h, m = dataset.get_random_train(args.batch_size) 
    gt = gt.to(device).float()
    h = h.to(device).float()
    m = m.to(device).float()

    output = net(m, h)
    loss = criterion(output, gt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end = time.time()
    
    print(f"epoch:{epoch}, time: {time.time()-start}, loss: {loss.item()}")    
    lr_scheduler.step()

eval(net, dataset, epoch)

checkpoint = {
'model': net.state_dict(),
'optimizer': optimizer.state_dict(),
}
model_path = os.path.join(args.exp_name, 'model', 'checkpoint.pth.tar')
torch.save(checkpoint, model_path)
