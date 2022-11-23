from pdf2image import convert_from_path
from tqdm import tqdm
import os, sys
from PIL import Image
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

def convert_pdf_to_images(pdf_path):
    save_dir = "/".join(pdf_path.split("/")[:-1])
    img_prefix = pdf_path.split("/")[-1]
    img_prefix, img_type = ".".join(img_prefix.split(".")[:-1]), img_prefix.split(".")[-1]
    if img_type.lower() == "pdf":
        if not os.path.exists(f"{save_dir}/{img_prefix}.png"):
            images = convert_from_path(pdf_path)
            if len(images) != 1:
                raise ValueError("Should be 1 image in the pdf")
            image = images[0]
            image.save(f"{save_dir}/{img_prefix}.png")
    elif img_type.lower() in ["tif", "tiff"]:
        if not os.path.exists(f"{save_dir}/{img_prefix}.png"):
            img = Image.open(pdf_path)
            try:
                rgb_img = img.convert("RGB")
            except OSError:
                print(pdf_path)
                print("THIS FILE IS UNSAVEABLE, REMOVE IT MANUALLY AND IGNORE")
                sys.exit()
            rgb_img.save(f"{save_dir}/{img_prefix}.png")
