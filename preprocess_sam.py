import argparse
from fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
import os
import time
import pickle as pkl
from tqdm import tqdm
import numpy as np
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM-x.pt", help="model"
    )
    parser.add_argument(
        "--img_folder_path", type=str, default="./test_unit/test_speed100", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
            "cuda:4"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()



def parser_filename(filename):
    parts = os.path.splitext(filename)[0].split('_')
    # 检查分割后的部分是否符合预期
    if len(parts) == 3:
        subject, action_cam, frameidx = parts
        action = action_cam.split('.')[0]
        cam = action_cam.split('.')[1]
        return subject, action, cam, frameidx
    elif len(parts) == 4:
        subject, action, idx_cam, frameidx = parts
        idx = idx_cam.split('.')[0]
        cam = idx_cam.split('.')[1]
        action = action + ' ' + str(idx)
        return subject, action, cam, frameidx
    


"""
    input:
        S*/ image 
    image: subject_action.cam_frameidx.jpg
    output:
        dict:
            S*:
                action*:
                    cam:
                        frame_idx
"""
def main(args):
    # load model
    saveDic = {}
    model = FastSAM(args.model_path)
    start_time = time.time()
    for file in tqdm(os.listdir(args.img_folder_path)):
        img_path = os.path.join(args.img_folder_path, file)
        input = Image.open(img_path)
        input = input.convert("RGB")
        everything_results = model(
            input,
            device=args.device,
            retina_masks=args.retina,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou
            )
        prompt_process = FastSAMPrompt(input, everything_results, device=args.device)
        # get output mask 
        ann = prompt_process.everything_prompt()
        with open("mask1.pkl", 'wb') as f:
            pkl.dump(ann[0].cpu().numpy(), f)
        # save data
        subject, action, cam, frame_idx = parser_filename(file)

    #     if subject not in saveDic.keys():
    #         saveDic[subject] = {}
    #     if action not in saveDic[subject].keys():
    #         saveDic[subject][action] = {}
    #     if cam not in saveDic[subject][action].keys():
    #         saveDic[subject][action][cam] = {}     
    #     saveDic[subject][action][cam][frame_idx] = {}
    #     saveDic[subject][action][cam][frame_idx] = ann

    # end_time = time.time()
    # print("time cost {}".format(end_time - start_time))
    # with open('seg_dic.pkl', 'wb') as f:
    #     pkl.dump(saveDic, f)

        # prompt_process.plot(
        #     annotations=ann,
        #     output_path=args.output+img_path.split("/")[-1],
        #     bboxes = bboxes,
        #     points = points,
        #     point_label = point_label,
        #     withContours=args.withContours,
        #     better_quality=args.better_quality,
        # )




if __name__ == "__main__":
    args = parse_args()
    main(args)
    
