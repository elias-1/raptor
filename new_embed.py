import json
import os, sys
import argparse
from io import BytesIO

import numpy as np
from tqdm import tqdm
import torch
import zipfile
from time import time
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

LATENT_SIZE_LOOKUP = dict(
    SAM=256,
    MedSAM=256,
    CLIP=1024,
    DINOv2=1024,
    DINOv3=1024,
    LlavaMed=1024,
)

def get_image_encoder(encoder_name, device):

    preprocessor = lambda imgs: imgs
    if encoder_name in ['MedSAM', 'SAM']:
        sys.path.append('../etc/MedSAM')
        from segment_anything import sam_model_registry
        sam_checkpoints = dict(
            MedSAM='../etc/MedSAM/work_dir/MedSAM/medsam_vit_b.pth',
            SAM='../etc/MedSAM/work_dir/MedSAM/sam_vit_b_01ec64.pth',
        )
        image_encoder = sam_model_registry['vit_b'](checkpoint=sam_checkpoints[encoder_name])

    elif encoder_name in ['DINOv2', 'DINOv3', 'CLIP']:
        model_url = dict(
            DINOv2='facebook/dinov2-large',
            DINOv3='facebook/dinov3-vitl16-pretrain-lvd1689m',
            CLIP='openai/clip-vit-large-patch14'
        )[encoder_name]

        processor = AutoImageProcessor.from_pretrained(model_url, use_fast=True)
        image_encoder = AutoModel.from_pretrained(model_url).to(device)

        if encoder_name == 'CLIP':
            image_encoder = image_encoder.vision_model

        preprocessor = lambda imgs: processor(images=[Image.fromarray((i*256).astype(np.uint8)) for i in imgs], return_tensors="pt").to(device)

    elif encoder_name == 'LlavaMed':
        sys.path.append('../etc/LLaVA-Med')
        from llava.model.builder import load_pretrained_model
        tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path='/u/scratch/u/ulzee/hug/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/f2f72301dc934e74948b5802c87dbc83d100e6bd/',
                model_base=None,
                model_name='llava-med-v1.5-mistral-7b'
        )
        image_encoder = model.get_vision_tower()

    else:
        raise 'Unknown model'

    return preprocessor, image_encoder

def crop_pad_matrix(mat, size=224):
    if all([dim == size for dim in mat.shape]):
        return mat

    mat = torch.nn.functional.interpolate(
        torch.from_numpy(mat[np.newaxis, np.newaxis, :, :]), size=(size, size),
        mode='bicubic'
    ).squeeze().numpy()
    return mat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--folder', type=str, default=None, help='Folder containing scans in .nii.gz or .zip (UKBB style)')
    parser.add_argument('--extract_file', type=str, default='T1/T1_brain.nii.gz', help='The nii.gz to read if given zip files')
    parser.add_argument('--encoder', type=str, default='DINOv2', help='Encoder type')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--num_torch_threads', type=int, default=4, help='Batch size')
    parser.add_argument('--saveto', type=str, required=True, help='Save directory')
    parser.add_argument('--k', type=str, default=None)
    parser.add_argument('--planes', default='A,C,S')
    parser.add_argument('--avgpool', default=False, action='store_true')

    args = parser.parse_args()

    torch.set_num_threads(args.num_torch_threads)

    args.planes = args.planes.split(',')
    for p in args.planes:
        assert p in 'ACS'
    if args.k is not None:
        args.k = args.k.split(',')
        print(f'Loading {len(args.k)} projections...')
    if args.avgpool:
        assert args.k is None # there is no need for Ks
    else:
        assert args.k is not None

    with open(os.path.join(os.path.dirname(args.folder), "train_val_test2.json"), 'r') as f:
        train_val_test = json.load(f)

    fbatch = []
    for split in ['train', 'val', 'test']:
        patient_info = train_val_test[split]

        for patient in patient_info:
            cur_zip_paths = []
            for filename in patient_info[patient]['filenames']:
                cur_zip_paths.append(os.path.join(args.folder, filename.split('_')[0], patient + "_" + filename + '.zip'))
            fbatch.extend(cur_zip_paths)

    preprocessor, image_encoder = get_image_encoder(args.encoder, args.device)
    image_encoder = image_encoder.to(args.device).eval()

    projfiles = []
    if args.k is not None:
        for fname in args.k:
            projfiles += [(fname.split('/')[-1].split('.')[0], np.load(fname))]
    else:
        projfiles = [('proj_identity', None)]

    pbar = tqdm(fbatch)
    for file_path in pbar:
        pid = os.path.basename(file_path)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            filenames = sorted(zip_ref.namelist(), key=lambda x: int(x.split('_')[-1].split('.')[0]))

            slices = []
            for filename in filenames:
                slice_data = zip_ref.read(filename)
                f = BytesIO(slice_data)
                image = Image.open(f)
                slices.append(np.array(image))

        vol = np.stack(slices)
        vol = vol.astype(float)
        vol /= 256

        # collect slices (in axes order)
        slices_byxyz = []

        ntot = []
        if 'A' in args.planes:
            slices = []
            for i in range(0, vol.shape[0]-0):
                if vol[i].shape[0] == 1 or vol[i].shape[1] == 1: continue
                slices += [vol[i]]
            nx = len(slices)
            ntot += [nx]
            slices_byxyz += slices

        if 'C' in args.planes:
            slices = []
            for i in range(0, vol.shape[1]-0):
                if vol[:, i].shape[0] == 1 or vol[:, i].shape[1] == 1: continue
                slices += [vol[:, i]]
            ny = len(slices)
            ntot += [ny]
            slices_byxyz += slices

        if 'S' in args.planes:
            slices = []
            for i in range(0, vol.shape[2]-0):
                if vol[:, :, i].shape[0] == 1 or vol[:, :, i].shape[1] == 1: continue
                slices += [vol[:, :, i]]
            nz = len(slices)
            ntot += [nz]
            slices_byxyz += slices

        t0 = time()
        imgs = [crop_pad_matrix(img) for img in slices_byxyz]
        t_crop = time() - t0
        embs = []
        t0 = time()
        for i in range(0, len(imgs), args.batch_size):
            inputs = preprocessor(imgs[i:i+args.batch_size])

            with torch.no_grad():
                outputs = image_encoder(**inputs)
                last_hidden_states = outputs.last_hidden_state
                embs += [e.T for e in last_hidden_states.detach().cpu().numpy()]

        savetimes = []
        for (projname, projmat) in projfiles:
            t0 = time()
            if projmat is not None:
                if len(args.planes) < 3:
                    projname = f'p{"".join(args.planes)}_{projname}'

                # projmat: D x K         (D: ViT dim, K: projections)
                # embs: S x D x 16 x 16  (S: slices)
                assert len(projmat) == len(embs[0])

            if not os.path.exists(f'{args.saveto}/{projname}'):
                os.makedirs(f'{args.saveto}/{projname}')

            # proj_embs: slices (16 + 16 + 16) x K x 256
            # proj_embs_sum: slices 3 x K x 16 x 16
            assert np.sum(ntot) == len(embs)

            if args.avgpool:
                proj_embs = proj_embs.reshape(len(proj_embs), LATENT_SIZE_LOOKUP[args.encoder], -1)
                proj_embs_sum_flat = proj_embs.mean((0, -1))
            else:
                plane_breaks = []
                agg = 0
                for s_count in ntot:
                    plane_breaks += [s_count + agg]
                    agg += s_count

                byside = [s for s in np.split(embs, plane_breaks, axis=0) if len(s)]
                byside = [s.mean(0) for s in byside]
                byside = [projmat.T @ s.reshape(projmat.shape[0], -1) for s in byside]
                assert len(byside) == len(args.planes)
                proj_embs_sum = np.concatenate(byside)

                # proj_embs_sum: slices 3K x 256 ~ 7680 for K=10
                proj_embs_sum_flat = proj_embs_sum.reshape(-1)

            t0 = time()
            pid_save_name = pid.split('.')[0] # strip any extensions
            np.save(f'{args.saveto}/{projname}/{pid_save_name}.npy', proj_embs_sum_flat.astype(np.float32))
            savetimes += [time() - t0]

        pbar.set_postfix(dict(
            pid=pid, sh=vol.shape[0], ns=len(slices_byxyz), d=proj_embs_sum_flat.shape,
        ))
