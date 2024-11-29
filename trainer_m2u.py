from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F
# from apex import amp
import random

def prompt_and_decoder(args, batched_input, model, image_embeddings, modality_embeddings, decoder_iter = False):
    # prompt
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None
    
    # decode
    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )
            dense_embeddings += modality_embeddings

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )
        dense_embeddings += modality_embeddings
    

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )
    
    # process output
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions

def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key =='image_m1' or key == 'image_m2' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def train_sam_pipeline(args, model, optimizer, batched_input, criterion, modality):
    """
    For main pipeline training
    """
    batched_input = stack_dict_batched(batched_input)
    batched_input = to_device(batched_input, args.device)
    model = model["sam_model"]
    modality_encoder2 = model["modality_encoder2"]
    
    labels = batched_input["label"]
    if modality == "m1":
        images = batched_input["image_m1"]
        aux_images = batched_input["image_m2"]
    elif modality == "m2":
        images = batched_input["image_m2"]
        aux_images = batched_input["image_m1"]
    
    if random.random() > 0.5:
        batched_input["point_coords"] = None
        flag = "boxes"
    else:
        batched_input["boxes"] = None
        flag = "point"

    for n, value in model.image_encoder.named_parameters():
        if "Adapter" in n:
            value.requires_grad = True
        else:
            value.requires_grad = False

    def repeat_embeddings(image_embeddings):
        B, _, _, _ = image_embeddings.shape
        image_embeddings_repeat = []
        for i in range(B):
            image_embed = image_embeddings[i]
            image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
            image_embeddings_repeat.append(image_embed)
        image_embeddings = torch.cat(image_embeddings_repeat, dim=0)
        return image_embeddings
    
    if args.use_amp:
        labels = labels.half()
        image_embeddings = model.image_encoder(images.half())
        # only m1 sam pipeline need m2 modality embeddings
        modality_embeddings = modality_encoder_2(aux_images.half()) if modality == "m1" else None

        image_embeddings = repeat_embeddings(image_embeddings)
        masks, low_res_masks, iou_predictions = prompt_and_decoder(args, 
                                                                    batched_input, 
                                                                    model, 
                                                                    image_embeddings, 
                                                                    modality_embeddings,
                                                                    decoder_iter = False)
        
        loss = criterion(masks, labels, iou_predictions)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(retain_graph=False)

    else:
        labels = batched_input["label"]
        image_embeddings = model.image_encoder(images) # B, 256, 16, 16
        # only m1 sam pipeline need m2 modality embeddings
        modality_embeddings = modality_encoder_2(aux_images) if modality == "m1" else None
        
        image_embeddings = repeat_embeddings(image_embeddings)
        masks, low_res_masks, iou_predictions = prompt_and_decoder(args, 
                                                                    batched_input, 
                                                                    model, 
                                                                    image_embeddings, 
                                                                    modality_embeddings, 
                                                                    decoder_iter = False)
        loss = criterion(masks, labels, iou_predictions)
        loss.backward(retain_graph=False)

    optimizer.step()
    optimizer.zero_grad()

    point_num = random.choice(args.point_list)
    batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
    batched_input = to_device(batched_input, args.device)

    image_embeddings = image_embeddings.detach().clone()
    for n, value in model.named_parameters():
        if "image_encoder" in n:
            value.requires_grad = False
        else:
            value.requires_grad = True

    init_mask_num = np.random.randint(1, args.iter_point - 1)
    for iter in range(args.iter_point):
        if iter == init_mask_num or iter == args.iter_point - 1:
            batched_input = setting_prompt_none(batched_input)

        if args.use_amp:
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=True)
            loss = criterion(masks, labels, iou_predictions)
            with amp.scale_loss(loss,  optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        else:
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=True)
            loss = criterion(masks, labels, iou_predictions)
            loss.backward(retain_graph=True)
            
        optimizer.step()
        optimizer.zero_grad()
        
        if iter != args.iter_point - 1:
            point_num = random.choice(args.point_list)
            batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
            batched_input = to_device(batched_input, args.device)
    
        if int(batch+1) % 50 == 0:
            if iter == init_mask_num or iter == args.iter_point - 1:
                print(f'Epoch: {epoch+1}, Batch: {batch+1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}')
            else:
                print(f'Epoch: {epoch+1}, Batch: {batch+1}, point {point_num} prompt: { SegMetrics(masks, labels, args.metrics)}')

    if int(batch+1) % 200 == 0:
        print(f"epoch:{epoch+1}, iteration:{batch+1}, loss:{loss.item()}")
        save_path = os.path.join(f"{args.work_dir}/models", args.run_name, f"epoch{epoch+1}_batch{batch+1}_sam.pth")
        state = {'model': model.state_dict(), 'optimizer': optimizer}
        torch.save(state, save_path)

    train_batch_metrics = SegMetrics(masks, labels, args.metrics)

    return train_batch_metrics, loss


def train_disentangle_pipeline(args, model, optimizer, batched_input):
    # model process
    modality_encoder_1 = model["modality_encoder_1"]
    modality_encoder_2 = model["modality_encoder_2"]
    rec_decoder = model["rec_decoder"]
    
    # input process
    batched_input = stack_dict_batched(batched_input)
    batched_input = to_device(batched_input, args.device)
    
    labels = batched_input["label"]
    images_1 = batched_input["image_m1"]
    images_2 = batched_input["image_m2"]

    if args.use_amp:
        image_embeddings_1 = model.image_encoder(images_2.half())
        image_embeddings_2 = model.image_encoder(images_2.half())
        modality_embeddings_1 = modality_encoder_1(images_1.half())
        modality_embeddings_2 = modality_encoder_2(images_2.half())
        
        # compute loss1 
        similarity = torch.mean(F.cosine_similarity(image_embeddings_1.view(B, -1), image_embeddings_2.view(B, -1), dim=1))
        loss1 = 1 - similarity
        
        # compute loss2
        loss2 = torch.mean(F.cosine_similarity(modality_embeddings_1.view(B, -1), modality_embeddings_2.view(B, -1), dim=1))
        
        # compute loss3
        def batch_cos_similarity(batch):
            """
            计算一个batch中每对样本的余弦相似度的均值
            """
            dot_product = torch.mm(batch, batch.t())
            norms = torch.norm(batch, dim=1)
            cosine_similarity = dot_product / torch.ger(norms, norms)
        return torch.mean(cosine_similarity)
        loss3 = 1 - ((batch_cos_similarity(modality_embeddings_1) + batch_cos_similarity(modality_embeddings_1)) / 2)
        loss_sim = loss1 + loss2 + loss3
        
        # compute rec loss
        image_embeddings_1_hat = image_embeddings_2 + modality_embeddings_1
        image_embeddings_2_hat = image_embeddings_1 + modality_embeddings_2
        image_1_hat, loss_rec_1 = rec_decoder(image_1, image_embeddings_1_hat)
        image_2_hat, loss_rec_2 = rec_decoder(image_2, image_embeddings_2_hat)
        loss_rec = loss_rec_1 + loss_rec_2
        
        # compute dis loss
        loss = loss_rec + loss_sim
        
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(retain_graph=False)

    else:
        image_embeddings_1 = model.image_encoder(images_2)
        image_embeddings_2 = model.image_encoder(images_2)
        modality_embeddings_1 = modality_encoder_1(images_1)
        modality_embeddings_2 = modality_encoder_2(images_2)
        
        # compute loss1 
        similarity = torch.mean(F.cosine_similarity(image_embeddings_1.view(B, -1), image_embeddings_2.view(B, -1), dim=1))
        loss1 = 1 - similarity
        
        # compute loss2
        loss2 = torch.mean(F.cosine_similarity(modality_embeddings_1.view(B, -1), modality_embeddings_2.view(B, -1), dim=1))
        
        # compute loss3
        def batch_cos_similarity(batch):
            """
            计算一个batch中每对样本的余弦相似度的均值
            """
            dot_product = torch.mm(batch, batch.t())
            norms = torch.norm(batch, dim=1)
            cosine_similarity = dot_product / torch.ger(norms, norms)
        return torch.mean(cosine_similarity)
        loss3 = 1 - ((batch_cos_similarity(modality_embeddings_1) + batch_cos_similarity(modality_embeddings_1)) / 2)
        loss_sim = loss1 + loss2 + loss3
        
        # compute rec loss
        image_embeddings_1_hat = image_embeddings_2 + modality_embeddings_1
        image_embeddings_2_hat = image_embeddings_1 + modality_embeddings_2
        image_1_hat, loss_rec_1 = rec_decoder(image_1, image_embeddings_1_hat)
        image_2_hat, loss_rec_2 = rec_decoder(image_2, image_embeddings_2_hat)
        loss_rec = loss_rec_1 + loss_rec_2
        
        # compute dis loss
        loss = loss_rec + loss_sim
        
        loss.backward(retain_graph=False)

    optimizer.step()
    optimizer.zero_grad()
    
    return loss_rec, loss_sim
    