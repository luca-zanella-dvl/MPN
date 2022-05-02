import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use("Agg")
from model.utils import DataLoader
from model.base_model import *
from utils import *
from tqdm import tqdm
import argparse
import warnings
import wandb
wandb.init(project="MPN")
warnings.filterwarnings("ignore") 




parser = argparse.ArgumentParser(description="MPN")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--loss_fra_reconstruct', type=float, default=1.00, help='weight of the frame reconstruction loss')
parser.add_argument('--loss_fea_reconstruct', type=float, default=1.00, help='weight of the feature reconstruction loss')
parser.add_argument('--loss_distinguish', type=float, default=0.0001, help='weight of the feature distinction loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr_D', type=float, default=1e-4, help='initial learning rate for parameters')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--segs', type=int, default=32, help='num of video segments')
parser.add_argument('--fdim', type=list, default=[128], help='channel dimension of the features')
parser.add_argument('--pdim', type=list, default=[128], help='channel dimension of the prototypes')
parser.add_argument('--psize', type=int, default=10, help='number of the prototype items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='.data/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--resume', type=str, default='exp/ped2/example.pth', help='file path of resume pth')
parser.add_argument('--debug', type=bool, default=False, help='if debug')
parser.add_argument('--val', action='store_true', help='if val')
args = parser.parse_args()

torch.manual_seed(2020)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus[0]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance


# Loading dataset
train_folder = os.path.join(args.dataset_path, args.dataset_type, "train/frames")
train_dataset = DataLoader(train_folder, transforms.Compose([
            transforms.ToTensor(),           
            ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)
train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
train_size = len(train_dataset)

if args.val is True: 
  val_folder = os.path.join(args.dataset_path, args.dataset_type, "val/frames")
  val_dataset = DataLoader(val_folder, transforms.Compose([
              transforms.ToTensor(),           
              ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

  
  val_size =len(val_dataset)
  val_batch = data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)



# Model setting
model = convAE(args.c, args.t_length, args.psize, args.fdim[0], args.pdim[0])
model.cuda()

params_encoder =  list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params_proto = list(model.prototype.parameters())
params_output = list(model.ohead.parameters())
params_D =  params_encoder+params_decoder+params_output+params_proto

optimizer_D = torch.optim.Adam(params_D, lr=args.lr_D)

start_epoch = 0
if os.path.exists(args.resume):
  print('Resume model from '+ args.resume)
  ckpt = args.resume
  checkpoint = torch.load(ckpt)
  start_epoch = checkpoint['epoch']
  model.load_state_dict(checkpoint['state_dict'].state_dict())
  optimizer_D.load_state_dict(checkpoint['optimizer_D'])

if len(args.gpus[0])>1:
  model = nn.DataParallel(model)

# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
if not args.debug:
  orig_stdout = sys.stdout
  f = open(os.path.join(log_dir, 'log.txt'),'w')
  sys.stdout= f



loss_func_mse = nn.MSELoss(reduction='none')
loss_pix = AverageMeter()
loss_fea = AverageMeter()
loss_dis = AverageMeter()

loss_pix_v = AverageMeter()
loss_fea_v = AverageMeter()
loss_dis_v = AverageMeter()

# Training

model.train()

for epoch in range(start_epoch, args.epochs):
    labels_list = []
    model.train() 
    pbar = tqdm(total=len(train_batch))
    for j,(imgs,_) in enumerate(train_batch):
        imgs = Variable(imgs).cuda()
        imgs = imgs.view(args.batch_size,-1,imgs.shape[-2],imgs.shape[-1])

        outputs, _, _, _, fea_loss, _, dis_loss = model.forward(imgs[:,0:12], None, True)
        optimizer_D.zero_grad()
        loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:,12:]))
        fea_loss = fea_loss.mean()
        dis_loss = dis_loss.mean()
        loss_D = args.loss_fra_reconstruct*loss_pixel + args.loss_fea_reconstruct * fea_loss + args.loss_distinguish * dis_loss 
        loss_D.backward(retain_graph=True)
        optimizer_D.step()


        loss_pix.update(args.loss_fra_reconstruct*loss_pixel.item(),  1)
        loss_fea.update(args.loss_fea_reconstruct*fea_loss.item(),  1)
        loss_dis.update(args.loss_distinguish*dis_loss.item(),  1)

        pbar.set_postfix({
                      'Epoch': '{0} {1}'.format(epoch+1, args.exp_dir),
                      'Lr': '{:.6f}'.format(optimizer_D.param_groups[-1]['lr']),
                      'PRe': '{:.6f}({:.6f})'.format(loss_pixel.item(), loss_pix.avg),
                      'FRe': '{:.6f}({:.6f})'.format(fea_loss.item(), loss_fea.avg),
                      'Dist': '{:.6f}({:.6f})'.format(dis_loss.item(), loss_dis.avg),
                    })
        pbar.update(1)

    print('----------------------------------------')
    print('Epoch:', epoch+1)
    print('Lr: {:.6f}'.format(optimizer_D.param_groups[-1]['lr']))
    print('PRe: {:.6f}({:.6f})'.format(loss_pixel.item(), loss_pix.avg))
    print('FRe: {:.6f}({:.6f})'.format(fea_loss.item(), loss_fea.avg))
    print('Dist: {:.6f}({:.6f})'.format(dis_loss.item(), loss_dis.avg))
    print('----------------------------------------')   
    
    
    
    pbar.close()  

    # Validation 
    if args.val: 
      model.eval()
      with torch.no_grad():
        for k,(imgs,_) in enumerate(val_batch):
            imgs = Variable(imgs).cuda()
            outputs, _, _, _, fea_loss_v, _, dis_loss_v = model.forward(imgs[:,0:12], None, True)

            loss_pixel_v = torch.mean(loss_func_mse(outputs, imgs[:,12:]))
            fea_loss_v = fea_loss_v.mean()
            dis_loss_v = dis_loss_v.mean()

            loss_pix_v.update(args.loss_fra_reconstruct*loss_pixel_v.item(),  1)
            loss_fea_v.update(args.loss_fea_reconstruct*fea_loss_v.item(),  1)
            loss_dis_v.update(args.loss_distinguish*dis_loss_v.item(),  1)
    
      print('----------------------------------------')
      print('Epoch:', epoch+1)
      print('PRe: {:.6f}({:.6f})'.format(loss_pixel_v.item(), loss_pix_v.avg))
      print('FRe: {:.6f}({:.6f})'.format(fea_loss_v.item(), loss_fea_v.avg))
      print('Dist: {:.6f}({:.6f})'.format(dis_loss_v.item(), loss_dis_v.avg))
      print('----------------------------------------')   
    
    wandb.define_metric("epoch")
    wandb.define_metric("Loss/*", step_metric="epoch")
    wandb.log({"Loss/Distinction": loss_dis.avg , "epoch": epoch + 1})
    wandb.log({"Loss/Feature-Reconstruction": loss_fea.avg, "epoch": epoch + 1})
    wandb.log({"Loss/Frame-Reconstruction": loss_pix.avg, "epoch": epoch +1})
    loss_pix.reset()
    loss_fea.reset()
    loss_dis.reset()
    
    if args.val: 
      
      wandb.log({"Loss/Distinction_V": loss_dis_v.avg, "epoch": epoch +1})
      wandb.log({"Loss/Feature-Reconstruction_V": loss_fea_v.avg, "epoch": epoch +1})
      wandb.log({"Loss/Frame-Reconstruction_V": loss_pix_v.avg, "epoch": epoch +1})
      loss_pix_v.reset()
      loss_fea_v.reset()
      loss_dis_v.reset()
   
    
  
    
    # Save the model
    if epoch%10==0:
      
      if len(args.gpus[0])>1:
        model_save = model.module
      else:
        model_save = model
        
      state = {
            'epoch': epoch,
            'state_dict': model_save.state_dict(),
            'optimizer_D' : optimizer_D.state_dict(),
          }
      torch.save(state, os.path.join(log_dir, 'model_'+str(epoch)+'.pth'))

    
print('Training is finished')
if len(args.gpus[0])>1:
  model_save = model.module
else:
  model_save = model
  
state = {
      'epoch': epoch,
      'state_dict': model_save.state_dict(),
      'optimizer_D' : optimizer_D.state_dict(),
    }
# Save the model
torch.save(state, os.path.join(log_dir, 'model_'+str(epoch)+'.pth'))

if not args.debug:
  f.close()

wandb.finish()
