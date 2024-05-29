import argparse
import os
import sys
import cv2
import json
import torch
import logging
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor


warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")

logging.basicConfig(filename='processed_folders.log', level=logging.INFO, format='%(asctime)s - %(message)s')


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.to(device)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower().strip() + ("" if caption.endswith(".") else ".")
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    tokenized = model.tokenizer(caption)
    pred_phrases = [get_phrases_from_posmap(logit > text_threshold, tokenized, model.tokenizer) + (f"({str(logit.max().item())[:4]})" if with_logits else "")
                    for logit in logits_filt]
    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def plot_gs_output(image_cv, masks, boxes_filt, pred_phrases, output_dir, image_name_base):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_cv)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt.cpu(), pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis("off")

    plt.savefig(os.path.join(output_dir, f"{image_name_base}.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()


def save_mask_data(mask_jpg_out_dir, mask_json_out_dir, mask_list, box_list, label_list, mask_filename):
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis("off")

    plt.savefig(os.path.join(mask_jpg_out_dir, mask_filename + ".jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()
    json_data = [{"value": idx + 1, "label": label.split("(")[0], "logit": float(label.split("(")[1][:-1]), "box": box.numpy().tolist()} for idx, (label, box) in enumerate(zip(label_list, box_list))]
    with open(os.path.join(mask_json_out_dir, mask_filename + ".json"), "w") as f:
        json.dump(json_data, f)


def find_omni_folders(root_dir):
    omni_folders = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if 'omni' in dirnames and "calib" not in dirpath:
            omni_folders.append(os.path.join(dirpath, 'omni'))
            dirnames.clear()

    return omni_folders

def make_output_directories(folder_path):
    output_base_dir = os.path.join(str(Path(folder_path).parent), 'omni_masks')
    mask_jpg_out_dir = os.path.join(output_base_dir, 'out_mask') #  os.path.basename(image_path).split('.')[0]
    os.makedirs(mask_jpg_out_dir, exist_ok=True)

    mask_json_out_dir = os.path.join(output_base_dir, "out_json")
    os.makedirs(mask_json_out_dir, exist_ok=True)

    gs_out_composite_dir = os.path.join(output_base_dir, "out_composite")
    os.makedirs(gs_out_composite_dir, exist_ok=True)
    return mask_jpg_out_dir, mask_json_out_dir, gs_out_composite_dir


def process_images_in_folder(model, predictor, folder_path, text_prompt, box_threshold, text_threshold, device):
    logging.info(f"[*] Processing images in {folder_path}")
    mask_jpg_out_dir, mask_json_out_dir, gs_out_composite_dir = make_output_directories(folder_path)
    for image_name in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, image_name)
        if not image_path.lower().endswith(('jpg')):
            continue
        
        try:
            image_pil, image = load_image(image_path)
        except Exception as e:
            logging.error(f"[*] Error loading image: {image_path}")
            continue
        
        boxes_filt, pred_phrases = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, device=device)
        if boxes_filt.size(0) == 0:
            continue

        image_cv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        predictor.set_image(image_cv)
        size = image_pil.size
        boxes_filt = (boxes_filt * torch.tensor([size[0], size[1], size[0], size[1]])).to(device)
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
        boxes_filt[:, 2:] += boxes_filt[:, :2]

        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(device)
        masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)
        
        image_name_base = os.path.basename(image_path).split('.')[0]
        save_mask_data(mask_jpg_out_dir, mask_json_out_dir, masks, boxes_filt.cpu(), pred_phrases, image_name_base)
        # plot_gs_output(image_cv, masks, boxes_filt, pred_phrases, gs_out_composite_dir, image_name_base)
    logging.info(f"[*] Processed folder: {folder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")
    parser.add_argument("--input_dir", type=str, required=True, help="path to input directory")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=False, help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    config_file = args.config
    grounded_checkpoint = args.grounded_checkpoint
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    input_dir = args.input_dir
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # os.makedirs(output_dir, exist_ok=True)
    model = load_model(config_file, grounded_checkpoint, device=device)
    predictor = (SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device)) if use_sam_hq else
                 SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)))

    print("[*] Finding omni folders...")
    omni_folders = find_omni_folders(input_dir)

    for omni_folder in tqdm(omni_folders):
        process_images_in_folder(model, predictor, omni_folder, text_prompt, box_threshold, text_threshold, device)

    logging.shutdown()