
import torch
import torch.nn as nn
import argparse
import os
import time
import random
import numpy as np
import datetime
import torch.nn.functional as F

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import ImageEncoderViT, ImageDecoderViT
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from metrics import SegMetrics
from tqdm import tqdm
from trainer_m2u import train_sam_pipeline, train_disentangle_pipeline

# from apex import amp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="checkpoint", help="work dir")
    parser.add_argument("--run_name", type=str, default="sam-med2d", help="run model name")
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="checkpoint/pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--use_amp", type=bool, default=False, help="use amp")
    args = parser.parse_args()

    return args


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    # sam_model = model["sam_model"]
    # modality_encoder_1 = model["modality_encoder_1"]
    # modality_encoder_2 = model["modality_encoder_2"]
    # rec_decoder = model["rec_decoder"]
    
    train_loader = tqdm(train_loader)
    train_losses_sam = []
    train_iter_metrics = [0] * len(args.metrics)
    train_losses_rec = []
    train_losses_sim = []
    
    for batch, batched_input in enumerate(train_loader):
        # sam pipeline
        ## m1 sam pipeline
        train_batch_metrics, loss_m1 = train_sam_pipeline(args, model, optimizer, batched_input, criterion, modality="m1")
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]
        ## m2 sam pipeline
        train_batch_metrics, loss_m2 = train_sam_pipeline(args, model, optimizer, batched_input, criterion, modality="m2")
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]
        ##
        train_losses_sam.append(loss_m1.item() + loss_m2.item())
        
        # disentangle pipeline
        loss_rec, loss_sim = train_disentangle_pipeline(args, model, optimizer, batched_input)
        train_losses_rec.append(loss_rec.item())
        train_losses_sim.append(loss_sim.item())
        
        #
        gpu_info = {}
        gpu_info['gpu_name'] = args.device 
        train_loader.set_postfix(loss_seg=loss_m1.item() + loss_m2.item(),
                                 loss_rec=loss_rec.item(), 
                                 loss_sim=loss_sim.item(),
                                 gpu_info=gpu_info)

    return train_losses, train_iter_metrics, train_losses_rec, train_losses_sim


def main(args):
    # model init
    model = {}
    sam_model = sam_model_registry[args.model_type](args).to(args.device)
    modality_encoder_1 = ImageEncoderViT(img_size=256, depth=6, num_heads=3)
    modality_encoder_2 = ImageEncoderViT(img_size=256, depth=6, num_heads=3)
    rec_decoder = ImageDecoderViT(image_size=256, depth=2, num_heads=3)
    model["sam_model"] = sam_model
    model["modality_encoder_1"] = modality_encoder_1
    model["modality_encoder_2"] = modality_encoder_2
    model["rec_decoder"] = rec_decoder
                
    # optimizer init
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.5)
        print('*******Use MultiStepLR')
        
    # loss init
    criterion = FocalDiceloss_IoULoss()

    # mixed precision
    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        print("*******Mixed precision with Apex")
    else:
        print('*******Do not use mixed precision')

    # data loader init
    train_dataset = TrainingDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name = False)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=4)
    print('*******Train data:', len(train_dataset))   
    
    # log init
    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))
    
    # training
    best_loss = 1e10
    l = len(train_loader)
    for epoch in range(0, args.epochs):
        model.train()
        train_metrics = {}
        start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)
        
        # forward $ backward one epoch
        train_losses_sam, train_iter_metrics, train_losses_rec, train_losses_sim = train_one_epoch(args, 
                                                                                                   model, 
                                                                                                   optimizer, 
                                                                                                   train_loader, 
                                                                                                   epoch, 
                                                                                                   criterion)

        # lr scheduler
        if args.lr_scheduler is not None:
            scheduler.step()
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr

        # metrics & losses record
        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}
        average_loss_sam = np.mean(train_losses_sam)
        average_loss_rec = np.mean(train_losses_rec)
        average_loss_sim = np.mean(train_losses_sim)
        loggers.info(f"epoch: {epoch + 1}, \
                     lr: {lr}, \
                     loss_sam: {average_loss_sam:.4f}, \
                     metrics: {train_metrics}, \
                     loss_rec: {average_loss_rec:.4f}, \
                     loss_sim: {average_loss_sim:.4f}")
        
        # save model
        if average_loss_sam < best_loss:
            best_loss = average_loss
            save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch+1}_sam.pth")
            state = {'model': model.float().state_dict(), 'optimizer': optimizer}
            torch.save(state, save_path)
            if args.use_amp:
                model = model.half()

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))


if __name__ == '__main__':
    # args = parse_args()
    # main(args)
    
    
    modality_encoder_1 = ImageEncoderViT(img_size=256, depth=6, num_heads=8)
    modality_encoder_2 = ImageEncoderViT(img_size=256, depth=6, num_heads=8)
    rec_decoder = ImageDecoderViT(img_size=256, decoder_depth=2, decoder_num_heads=8)
    img = torch.randn(4, 3, 256, 256)
    x = modality_encoder_1(img)
    pred, loss = rec_decoder(img, x)
    print(loss.shape)


