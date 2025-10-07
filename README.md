# Swimming Pools in São Paulo

**Rafael Guimarães**

###### [LinkedIn](https://www.linkedin.com/in/rafa-bg/) • [GitHub](https://github.com/rafael-b-g) • [X/Twitter](https://x.com/rafa_b_g) • ​[rafa.b.g@icloud.com](mailto:rafa.b.g@icloud.com)

## TL;DR
I counted the number of outdoor swimming pools in the city of São Paulo using computer vision and aerial imagery. Aerial images are from [GeoSampa](https://geosampa.prefeitura.sp.gov.br/)’s 2020 orthophotos. A [YOLO11-Segmentation](https://docs.ultralytics.com/tasks/segment/) model was trained and used as the pool detector.

Check out [pools.iglu.website](https://pools.iglu.website) to see the results.

## Goals
- [x] Train a pool detector with mAP > 0.65
- [x] Count the total number of outdoor swimming pools in São Paulo (with correction and confidence interval) using this model
- [x] Count the total combined pool area
- [x] Create an interactive heat map showing pool density across São Paulo
- [ ] Work at Cloudwalk 😅

## Dataset
### Finding good images
To successfully train a pool detector, I first needed a good training dataset.

After searching for satellite imagery of São Paulo, I quickly realized satellite wouldn’t be a good solution: clouds were a problem, the good images were scarce, and (most importantly) the resolution was terrible.

The solution was aerial imagery (orthophotos). That’s when I found [GeoSampa](https://geosampa.prefeitura.sp.gov.br/), a platform created and maintained by São Paulo’s municipality. The images covered the entire city of São Paulo, were taken in 2020, had good resolution, and were free.

The only issue was programmatically downloading the images. GeoSampa’s API didn’t allow direct access to the images, only to other kinds of maps hosted on the platform.

With a bit of reverse engineering, I found a way to download the images with a script.

```
from pathlib import Path
import requests
import zipfile
import shutil
import subprocess
import os
import concurrent.futures
import threading
import logging
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------- CONFIGURE HERE ----------
IDS = [
    # Image IDs to download

    # "3314-214",
    # "3314-261",
    # "2326-133",
    # "3313-313",
]
OUTDIR = Path("images/originals")
GDAL_QUALITY = "95"     # JPEG quality
MAX_WORKERS = min(8, len(IDS))  # how many parallel tasks (downloads+extracts+conversions) to run
MAX_GDAL_PROCS = max(1, (os.cpu_count() or 2) - 1)  # limit concurrent gdal_translate processes
# ------------------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pool-pipeline")

# Session with simple retry strategy
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Semaphore to limit how many gdal_translate calls we do at once.
gdal_semaphore = threading.Semaphore(MAX_GDAL_PROCS)


def download_zip(id_):
    params = {
        "orig": "DownloadMapaArticulacao",
        "arq": f"ORTOFOTOS_2020_IRGB\\{id_}.zip",
        "arqTipo": "MAPA_ARTICULACAO",
    }
    dest = OUTDIR / f"{id_}.zip"

    # if file already exists and non-empty, reuse it (resume)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"{id_}: ZIP already downloaded -> {dest.name}")
        return dest

    logger.info(f"{id_}: downloading ZIP...")
    with session.get(
        "https://download.geosampa.prefeitura.sp.gov.br/PaginasPublicas/downloadArquivo.aspx",
        params=params,
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

    logger.info(f"{id_}: downloaded {dest.name} ({dest.stat().st_size} bytes)")
    return dest


def extract_jp2(zip_path, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{zip_path.stem}: extracting JP2 from {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.lower().endswith(".jp2"):
                target = dest_dir / Path(name).name  # flatten paths
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                logger.info(f"{zip_path.stem}: extracted {target.name}")
                return target
    logger.warning(f"{zip_path.stem}: no .jp2 file found inside {zip_path.name}")
    return None


def jp2_to_jpeg(jp2_path, jpg_path):
    jpg_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdal_translate",
        "-of", "JPEG",
        "-co", f"QUALITY={GDAL_QUALITY}",
        str(jp2_path),
        str(jpg_path),
    ]
    logger.info(f"{jp2_path.stem}: running gdal_translate -> {jpg_path.name}")
    subprocess.run(cmd, check=True)
    logger.info(f"{jp2_path.stem}: created {jpg_path.name}")


def process_id(id_):
    try:
        start_ts = time.time()
        logger.info(f"{id_}: START")

        # ZIP download
        zip_path = download_zip(id_)

        # JP2 file extraction
        jp2_dir = OUTDIR / id_
        jp2 = extract_jp2(zip_path, jp2_dir)
        if not jp2:
            raise FileNotFoundError("No JP2 found inside zip")

        # conversion JPEG 2000 -> JPEG (limit concurrency with semaphore)
        jpg = OUTDIR / f"{id_}.jpg"
        if jpg.exists() and jpg.stat().st_size > 0:
            logger.info(f"{id_}: {jpg.name} already exists, skipping gdal_translate")
        else:
            gdal_semaphore.acquire()
            try:
                jp2_to_jpeg(jp2, jpg)
            finally:
                gdal_semaphore.release()

        # cleanup
        aux_path = OUTDIR / f"{id_}.jpg.aux.xml"
        aux_path.unlink(missing_ok=True)
        jp2.unlink(missing_ok=True)
        shutil.rmtree(jp2_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)

        elapsed = time.time() - start_ts
        logger.info(f"{id_}: DONE in {elapsed:.1f}s")
        return (id_, True, None)
    except Exception as e:
        logger.exception(f"{id_}: ERROR -> {e}")
        return (id_, False, str(e))


def main():
    logger.info(f"Starting pipeline with {MAX_WORKERS} workers, max {MAX_GDAL_PROCS} concurrent gdal_translate")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(process_id, id_): id_ for id_ in IDS}
        for fut in concurrent.futures.as_completed(futures):
            id_ = futures[fut]
            try:
                id_, ok, err = fut.result()
                if ok:
                    logger.info(f"{id_}: SUCCESS")
                else:
                    logger.error(f"{id_}: FAILED -> {err}")
            except Exception as e:
                logger.exception(f"{id_}: unhandled exception -> {e}")

    logger.info("All tasks finished.")


if __name__ == "__main__":
    main()
```

With that, I downloaded a few images from all across the city to use as my dataset.

I decided to use the IRGB (“I” for “Infrared”) images instead of the normal RGB ones, to make it easier for the detector to distinguish between trees and pools (since many pools are surrounded by trees). The red coloring of vegetation on IRGB images is much “further away” (chromatically) from the blue of the pools, compared to the green vegetation on normal RGB images.

Here’s an example of RGB vs IRGB:

![IRGB_RGB](https://github.com/user-attachments/assets/a1d3ef5e-0b0e-4b13-b3ce-41dc94b1d394)

Vegetation is red on IRGB because plants strongly reflect infrared light, while other objects stay mostly the same color.

### Slicing
The images from GeoSampa are way too big (in width and height) to be directly used by the pool detector, so I used [SAHI](https://github.com/obss/sahi) to slice the images into smaller (1024 by 1024) tiles, with 20% overlap between them (so every pool shows up unsliced in at least one tile).

```
from sahi.slicing import slice_image
import pathlib

IMAGE_DIR = "images/originals"      # raw JPEGs
OUT_DIR = "images/sliced"     # where tiles will go
SLICE_H = SLICE_W = 1024
OVERLAP_H = OVERLAP_W = 0.2   # 20% overlap

for p in pathlib.Path(IMAGE_DIR).glob("*.jpg"):
    slice_image(
        image=str(p),
        output_dir=OUT_DIR,
        output_file_name=f"{p.stem}_tiles",
        slice_height=SLICE_H,
        slice_width=SLICE_W,
        overlap_height_ratio=OVERLAP_H,
        overlap_width_ratio=OVERLAP_W,
        out_ext=".jpg",
    )
```

### Annotating
With all the dataset images sliced, I uploaded them to [CVAT](https://www.cvat.ai) and [Roboflow](https://roboflow.com) to annotate them (I initially used CVAT, but then switched to Roboflow).

Because I wanted to calculate not only the total pool count but also the total pool area, I decided to annotate the pools using polygons, instead of simple bounding boxes. This would later allow me to train a segmentation version of YOLO and get the area of each pool during inference.

Most of the image tiles had no pools in them, but I didn’t remove any tiles (I kept them as negative samples). I decided this was a good way to teach the detector what pools *didn’t* look like, and make the dataset represent real-world scenarios more accurately.

Annotation was by far the most tedious part of the project. I manually annotated 1467 image tiles, drawing a polygon around every pool, which wasn’t exactly fun.

### Building the final dataset
The script below applies some processing to make sure pools that fall on the tile overlap areas are shown in both tiles’ annotations (during annotation, I tried to save time by only annotating each pool once, even if it appeared in more than one tile, which is why this post-processing is needed).

```
import json
from collections import defaultdict
from shapely.geometry import Polygon, box


INPUT_PATH = "images/annotated/COCO.json"

with open(INPUT_PATH, 'r') as f:
    coco = json.load(f)

tiles = coco['images']
orig_to_tiles = defaultdict(list)
tile_to_orig = {}
tile_to_offset = {}
tile_to_size = {}

for img in tiles:
    fname = img['file_name']
    if '_tiles_' not in fname:
        continue
    orig, rest = fname.split('_tiles_')
    coords_str = rest[:-4]
    coords = coords_str.split('_')
    tlx, tly, brx, bry = map(int, coords)
    orig_to_tiles[orig].append(img)
    tile_to_orig[img['id']] = orig
    tile_to_offset[img['id']] = (tlx, tly)
    tile_to_size[img['id']] = (brx - tlx, bry - tly)

orig_to_annos = defaultdict(list)
for anno in coco['annotations']:
    if anno['iscrowd'] != 0:
        continue  # Assume no RLE, skip if crowd
    tile_id = anno['image_id']
    orig = tile_to_orig[tile_id]
    tlx, tly = tile_to_offset[tile_id]
    if 'bbox' in anno:
        x, y, w, h = anno['bbox']
        anno['bbox'] = [x + tlx, y + tly, w, h]
    if 'segmentation' in anno:
        new_seg = []
        for seg in anno['segmentation']:
            new_s = []
            for i in range(0, len(seg), 2):
                new_s.append(seg[i] + tlx)
                new_s.append(seg[i + 1] + tly)
            new_seg.append(new_s)
        anno['segmentation'] = new_seg
    anno['image_id'] = orig  # Temporary
    orig_to_annos[orig].append(anno)

def is_duplicate(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return False
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = inter_area / (box1_area + box2_area - inter_area)
    iomin = inter_area / min(box1_area, box2_area)
    return iou > 0.5 or iomin > 0.8

def remove_duplicates(annos):
    annos.sort(key=lambda a: a['area'], reverse=True)
    kept = []
    for anno in annos:
        if any(is_duplicate(anno['bbox'], k['bbox']) for k in kept):
            continue
        kept.append(anno)
    return kept

new_images = []
image_id_counter = 1
tile_to_new_id = {}
for orig in sorted(orig_to_tiles.keys()):
    for tile in orig_to_tiles[orig]:
        new_img = tile.copy()
        new_img['id'] = image_id_counter
        tile_to_new_id[tile['id']] = image_id_counter
        new_images.append(new_img)
        image_id_counter += 1

new_annos = []
anno_id_counter = max((anno['id'] for anno in coco['annotations']), default=0) + 1
for orig, annos in orig_to_annos.items():
    dedup_annos = remove_duplicates(annos)
    for tile in orig_to_tiles[orig]:
        old_tile_id = tile['id']
        new_tile_id = tile_to_new_id[old_tile_id]
        tlx, tly = tile_to_offset[old_tile_id]
        tile_width, tile_height = tile_to_size[old_tile_id]
        tile_rect = box(tlx, tly, tlx + tile_width, tly + tile_height)
        for anno in dedup_annos:
            segs = anno['segmentation']
            if not segs:
                continue
            exterior = list(zip(segs[0][::2], segs[0][1::2]))
            holes = [list(zip(s[::2], s[1::2])) for s in segs[1:]]
            poly = Polygon(exterior, holes)
            orig_area = poly.area
            if orig_area <= 0:
                continue
            clipped = poly.intersection(tile_rect)
            if clipped.is_empty:
                continue
            clipped_area = clipped.area
            if clipped_area / orig_area < 0.1:
                continue
            new_seg = []
            if clipped.geom_type == 'Polygon':
                ext_coords = list(clipped.exterior.coords)[:-1]
                flat_ext = [coord for pt in ext_coords for coord in pt]
                new_seg.append(flat_ext)
                for interior in clipped.interiors:
                    int_coords = list(interior.coords)[:-1]
                    flat_int = [coord for pt in int_coords for coord in pt]
                    new_seg.append(flat_int)
            elif clipped.geom_type == 'MultiPolygon':
                for p in clipped.geoms:
                    ext_coords = list(p.exterior.coords)[:-1]
                    flat_ext = [coord for pt in ext_coords for coord in pt]
                    new_seg.append(flat_ext)
                    for interior in p.interiors:
                        int_coords = list(interior.coords)[:-1]
                        flat_int = [coord for pt in int_coords for coord in pt]
                        new_seg.append(flat_int)
            else:
                continue
            for seg in new_seg:
                for i in range(0, len(seg), 2):
                    seg[i] -= tlx
                    seg[i + 1] -= tly
                    seg[i] = max(0, min(tile_width, seg[i]))
                    seg[i + 1] = max(0, min(tile_height, seg[i + 1]))
            minx, miny, maxx, maxy = clipped.bounds
            minx -= tlx
            miny -= tly
            maxx -= tlx
            maxy -= tly
            minx = max(0, minx)
            miny = max(0, miny)
            maxx = min(tile_width, maxx)
            maxy = min(tile_height, maxy)
            bbox = [minx, miny, maxx - minx, maxy - miny]
            new_anno = {
                "id": anno_id_counter,
                "category_id": anno["category_id"],
                "iscrowd": anno["iscrowd"],
                "segmentation": new_seg,
                "image_id": new_tile_id,
                "area": clipped_area,
                "bbox": bbox
            }
            new_annos.append(new_anno)
            anno_id_counter += 1

new_coco = {
    "info": coco.get("info", {}),
    "licenses": coco.get("licenses", []),
    "images": new_images,
    "annotations": new_annos,
    "categories": coco.get("categories", [])
}

with open("resliced_coco.json", 'w') as f:
    json.dump(new_coco, f)
```

With the annotated image tiles in hand, I just had to split the dataset into “train” and “validation”, and convert the annotations from the COCO format to the YOLO format.

```
import os
import shutil
from collections import defaultdict
import random
from ultralytics.data.converter import convert_coco
import json
from pathlib import Path

# Constants
COCO_DIR = "images/annotated"  # Directory containing the COCO JSON annotation file(s)
IMAGES_DIR = "images/annotated"   # Directory containing the tile images
OUTPUT_DIR = "dataset_v2"   # Output directory for the YOLO dataset structure
TEMP_DIR = os.path.join(COCO_DIR, "temp_yolo")  # Temporary directory for YOLO conversion
VAL_PREFIXES_LIST = []

def main():
    try:
        # Delete TEMP_DIR if it already exists to prevent incrementing
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        
        # Convert COCO to YOLO format (it will create TEMP_DIR fresh)
        convert_coco(
            labels_dir=COCO_DIR,
            save_dir=TEMP_DIR,
            use_segments=True,
            cls91to80=False
        )
        
        # Determine where labels were saved
        labels_dir = os.path.join(TEMP_DIR, 'labels', 'COCO')
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(image_extensions)]
        
        # Group images by prefix (original image name before first underscore)
        groups = defaultdict(list)
        for img in image_files:
            prefix = img.split('_')[0]
            groups[prefix].append(img)
        
        # Load COCO JSON to count pools
        json_files = list(Path(COCO_DIR).glob('*.json'))
        if not json_files:
            raise ValueError("No COCO JSON file found")
        with open(json_files[0], 'r') as f:
            coco_data = json.load(f)
        
        categories = coco_data.get('categories', [])
        pool_cat = next((cat for cat in categories if 'pool' in cat['name'].lower()), None)
        if pool_cat:
            pool_id = pool_cat['id']
            count_only_pool = True
        else:
            count_only_pool = False  # Count all annotations if no 'pool' category found
        
        image_id_to_file = {img['id']: img['file_name'] for img in coco_data.get('images', [])}
        prefix_to_count = defaultdict(int)
        for ann in coco_data.get('annotations', []):
            if not count_only_pool or ('category_id' in ann and ann['category_id'] == pool_id):
                if 'image_id' in ann:
                    img_file = image_id_to_file.get(ann['image_id'])
                    if img_file:
                        prefix = img_file.split('_')[0]
                        prefix_to_count[prefix] += 1
        
        # Prepare group list with counts
        prefixes = list(groups.keys())
        group_list = []
        for prefix in prefixes:
            count = prefix_to_count.get(prefix, 0)
            group_list.append((prefix, count))
        
        # Separate positives and negatives
        positive_group_list = [(p, c) for p, c in group_list if c > 0]
        negative_prefixes = [p for p, c in group_list if c == 0]
        
        total_pools = sum(c for _, c in positive_group_list)
        total_images = len(image_files)
        
        if VAL_PREFIXES_LIST:
            val_prefixes = set(p for p in VAL_PREFIXES_LIST if p in prefixes)
            train_prefixes = set(prefixes) - val_prefixes
            test_prefixes = set()
        elif len(positive_group_list) == 0:
            train_prefixes = set(negative_prefixes)
            val_prefixes = set()
            test_prefixes = set()
        else:
            # Sort positives by pool count descending
            positive_group_list.sort(key=lambda x: x[1], reverse=True)
            
            # Greedy assignment for positives
            targets = {
                'train': 0.85 * total_pools,
                'val': 0.15 * total_pools,
                'test': 0.0 * total_pools
            }
            bins = {'train': [], 'val': [], 'test': []}
            current = {'train': 0, 'val': 0, 'test': 0}
            for prefix, count in positive_group_list:
                remaining = {k: targets[k] - current[k] for k in bins}
                assign_to = max(remaining, key=remaining.get)
                bins[assign_to].append(prefix)
                current[assign_to] += count
            
            train_prefixes = set(negative_prefixes + bins['train'])
            val_prefixes = set(bins['val'])
            test_prefixes = set(bins['test'])
        
        # Create output directory structure
        os.makedirs(os.path.join(OUTPUT_DIR, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'val', 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'val', 'labels'), exist_ok=True)
        if test_prefixes:
            os.makedirs(os.path.join(OUTPUT_DIR, 'test', 'images'), exist_ok=True)
            os.makedirs(os.path.join(OUTPUT_DIR, 'test', 'labels'), exist_ok=True)
        
        # Move files for train
        for prefix in train_prefixes:
            for img in groups[prefix]:
                # Move image
                src_img = os.path.join(IMAGES_DIR, img)
                dst_img = os.path.join(OUTPUT_DIR, 'train', 'images', img)
                shutil.move(src_img, dst_img)
                
                # Handle corresponding label
                label_file = img.rsplit('.', 1)[0] + '.txt'  # Use image name, change extension to .txt
                src_label = os.path.join(labels_dir, label_file)
                dst_label = os.path.join(OUTPUT_DIR, 'train', 'labels', label_file)
                if os.path.exists(src_label):
                    shutil.move(src_label, dst_label)
                else:
                    # Create empty annotation file for negative samples
                    open(dst_label, 'w').close()
        
        # Move files for val
        for prefix in val_prefixes:
            for img in groups[prefix]:
                # Move image
                src_img = os.path.join(IMAGES_DIR, img)
                dst_img = os.path.join(OUTPUT_DIR, 'val', 'images', img)
                shutil.move(src_img, dst_img)
                
                # Handle corresponding label
                label_file = img.rsplit('.', 1)[0] + '.txt'  # Use image name, change extension to .txt
                src_label = os.path.join(labels_dir, label_file)
                dst_label = os.path.join(OUTPUT_DIR, 'val', 'labels', label_file)
                if os.path.exists(src_label):
                    shutil.move(src_label, dst_label)
                else:
                    # Create empty annotation file for negative samples
                    open(dst_label, 'w').close()
        
        # Move files for test
        for prefix in test_prefixes:
            for img in groups[prefix]:
                # Move image
                src_img = os.path.join(IMAGES_DIR, img)
                dst_img = os.path.join(OUTPUT_DIR, 'test', 'images', img)
                shutil.move(src_img, dst_img)
                
                # Handle corresponding label
                label_file = img.rsplit('.', 1)[0] + '.txt'  # Use image name, change extension to .txt
                src_label = os.path.join(labels_dir, label_file)
                dst_label = os.path.join(OUTPUT_DIR, 'test', 'labels', label_file)
                if os.path.exists(src_label):
                    shutil.move(src_label, dst_label)
                else:
                    # Create empty annotation file for negative samples
                    open(dst_label, 'w').close()
        
        # Print distribution
        train_pools = sum(prefix_to_count[p] for p in train_prefixes)
        val_pools = sum(prefix_to_count[p] for p in val_prefixes)
        test_pools = sum(prefix_to_count[p] for p in test_prefixes if p in prefix_to_count)
        
        train_images = sum(len(groups[p]) for p in train_prefixes)
        val_images = sum(len(groups[p]) for p in val_prefixes)
        test_images = sum(len(groups[p]) for p in test_prefixes)
        
        print("Pool distribution:")
        if total_pools > 0:
            print(f"Train: {train_pools} ({train_pools / total_pools * 100:.1f}%)")
            print(f"Val: {val_pools} ({val_pools / total_pools * 100:.1f}%)")
            print(f"Test: {test_pools} ({test_pools / total_pools * 100:.1f}%)")
        else:
            print("Train: 0 (100%)")
            print("Val: 0 (0%)")
            print("Test: 0 (0%)")
        
        print("Image distribution:")
        print(f"Train: {train_images} ({train_images / total_images * 100:.1f}%)")
        print(f"Val: {val_images} ({val_images / total_images * 100:.1f}%)")
        print(f"Test: {test_images} ({test_images / total_images * 100:.1f}%)")
        
        # Create YAML file
        yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
        names = {i: cat['name'] for i, cat in enumerate(categories)}
        yaml_content = f"""
path: {OUTPUT_DIR}
train: train/images
val: val/images
"""
        if test_prefixes:
            yaml_content += "test: test/images\n"
        
        yaml_content += """
names:
"""
        for idx, name in names.items():
            yaml_content += f"  {idx}: {name}\n"
        
        with open(yaml_path, 'w') as yaml_file:
            yaml_file.write(yaml_content.strip())
    
    finally:
        # Clean up temporary directory
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)

if __name__ == '__main__':
    main()
```

You can find the final dataset [here](https://github.com/rafael-b-g/sp-pools/tree/main/dataset_v2).

## Training
### Choosing the model
Now that I had the dataset ready, I could move on to training.

I chose the YOLO11-Small (segmentation) model (from Ultralytics) to be the pool detector. It’s a well-performing, well-established, and easy-to-use computer vision model.

### Hyperparameter tuning
To establish a baseline for good model performance, I first trained a pool detector using my dataset directly on Roboflow. Roboflow doesn’t let you download the model weights on the free tier (and I wasn’t going to pay for it), so this was just a test to set the bar for what my model should look like.

Roboflow model performance:

| Metric    | Value |
|-----------|-------|
| mAP@50    | 88.1% |
| mAP@50:95 | 69.9% |
| Precision | 83.9% |
| Recall    | 79.9% |

So I turned to hyperparameter tuning to get better results from my training, using Ultralytics. I spent a while trying to get [RayTune](https://docs.ray.io/en/latest/tune/index.html) to work on multiple GPUs, which turned out to be very frustrating and a complete waste of time. I ended up using Ultralytics’ built-in tuner.

Because this was a very GPU-hungry process, running it on Google Colab would take ages, and I would have to babysit the entire tuning process to make sure Colab didn’t disconnect my runtime due to inactivity, which wasn’t a good option.

With a quick Google search, I found [Modal](https://modal.com/), which was giving $30 of free credits to new users. I ended up creating multiple accounts to get enough credits to run the entire hyperparameter tuning on 4 parallel GPUs. Not to mention, I could now run it during the night, since Modal doesn’t disconnect your runtime for idleness.

### Final model
After many hours of tuning and many manual experiments, I landed on a good model:

| Metric    | Value |
|-----------|-------|
| mAP@50    | 90.6% |
| mAP@50:95 | 75.8% |
| Precision | 84.9% |
| Recall    | 85.8% |

Training script:

```
from ultralytics import YOLO

model = YOLO("yolo11s-seg.pt")

model.train(
    data="/root/dataset_v2/data.yaml",
    batch=64,
    imgsz=1024,
    device=[0, 1, 2, 3],
    plots=True,
    epochs=100,
    patience=50,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
)
```

You can find the model weights [here](https://github.com/rafael-b-g/sp-pools/blob/main/model.pt).

## Results
It was finally time to count the pools.

When I started this project, I didn’t think it would be feasible to run the pool detector on all 4180 images of São Paulo (more than 230,000 image tiles, after slicing). However, after doing some simple math, I realized I could easily run the model on a single L40S GPU and have it go through all the images in about 4 hours (which was nothing compared to how long tuning took).

This would be much easier and simpler than sampling images and trying to extrapolate the pool count to the entire city, not to mention I could later build a much more accurate map showing pool density across São Paulo.

### Downloading the images
The first step was downloading all 4180 images from GeoSampa to my Modal notebook (after extracting all the image IDs from GeoSampa’s website). I included a few image IDs in the script below for demonstration, but there are many more.

```
from pathlib import Path
import requests
import zipfile
import shutil
import subprocess
import os
import concurrent.futures
import threading
import logging
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------- CONFIGURE HERE ----------
IDS = ["3241-3", "3236-1", "3213-3", "3213-4", "2242-1", "2242-2", "2242-3", "2242-4", "3213-1", "3213-2", "2122-2", "3234-4", "3233-1"]
OUTDIR = Path("/mnt/nimbus/images/originals")
GDAL_QUALITY = "95"  # JPEG quality
MAX_WORKERS = min(16, len(IDS))  # how many parallel tasks (downloads+extracts+conversions) to run
MAX_GDAL_PROCS = 16  # limit concurrent gdal_translate processes
# ------------------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pool-pipeline")

# Session with simple retry strategy
session = requests.Session()
retries = Retry(
    total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Semaphore to limit how many gdal_translate calls we do at once.
gdal_semaphore = threading.Semaphore(MAX_GDAL_PROCS)


def download_zip(id_):
    params = {
        "orig": "DownloadMapaArticulacao",
        "arq": f"ORTOFOTOS_2020_IRGB\\{id_}.zip",
        "arqTipo": "MAPA_ARTICULACAO",
    }
    dest = OUTDIR / f"{id_}.zip"

    # if file already exists and non-empty, reuse it (resume)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"{id_}: ZIP already downloaded -> {dest.name}")
        return dest

    logger.info(f"{id_}: downloading ZIP...")
    with session.get(
        "https://download.geosampa.prefeitura.sp.gov.br/PaginasPublicas/downloadArquivo.aspx",
        params=params,
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

    logger.info(f"{id_}: downloaded {dest.name} ({dest.stat().st_size} bytes)")
    return dest


def extract_jp2(zip_path, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{zip_path.stem}: extracting JP2 from {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.lower().endswith(".jp2"):
                target = dest_dir / Path(name).name  # flatten paths
                with z.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                logger.info(f"{zip_path.stem}: extracted {target.name}")
                return target
    logger.warning(f"{zip_path.stem}: no .jp2 file found inside {zip_path.name}")
    return None


def jp2_to_jpeg(jp2_path, jpg_path):
    jpg_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdal_translate",
        "-of",
        "JPEG",
        "-co",
        f"QUALITY={GDAL_QUALITY}",
        str(jp2_path),
        str(jpg_path),
    ]
    logger.info(f"{jp2_path.stem}: running gdal_translate -> {jpg_path.name}")
    subprocess.run(cmd, check=True)
    logger.info(f"{jp2_path.stem}: created {jpg_path.name}")


def process_id(id_):
    try:
        start_ts = time.time()
        logger.info(f"{id_}: START")

        # ZIP download
        zip_path = download_zip(id_)

        # JP2 file extraction
        jp2_dir = OUTDIR / id_
        jp2 = extract_jp2(zip_path, jp2_dir)
        if not jp2:
            raise FileNotFoundError("No JP2 found inside zip")

        # conversion JPEG 2000 -> JPEG (limit concurrency with semaphore)
        jpg = OUTDIR / f"{id_}.jpg"
        if jpg.exists() and jpg.stat().st_size > 0:
            logger.info(f"{id_}: {jpg.name} already exists, skipping gdal_translate")
        else:
            gdal_semaphore.acquire()
            try:
                jp2_to_jpeg(jp2, jpg)
            finally:
                gdal_semaphore.release()

        # cleanup
        aux_path = OUTDIR / f"{id_}.jpg.aux.xml"
        aux_path.unlink(missing_ok=True)
        jp2.unlink(missing_ok=True)
        shutil.rmtree(jp2_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)

        elapsed = time.time() - start_ts
        logger.info(f"{id_}: DONE in {elapsed:.1f}s")
        return (id_, True, None)
    except Exception as e:
        logger.exception(f"{id_}: ERROR -> {e}")
        return (id_, False, str(e))


def main():
    logger.info(
        f"Starting pipeline with {MAX_WORKERS} workers, and max {MAX_GDAL_PROCS} concurrent gdal_translate"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(process_id, id_): id_ for id_ in IDS}
        for fut in concurrent.futures.as_completed(futures):
            id_ = futures[fut]
            try:
                id_, ok, err = fut.result()
                if ok:
                    logger.info(f"{id_}: SUCCESS")
                else:
                    logger.error(f"{id_}: FAILED -> {err}")
            except Exception as e:
                logger.exception(f"{id_}: unhandled exception -> {e}")

    logger.info("All tasks finished.")


main()
```

3 images seemed to be corrupted or unavailable (I’m not sure what happened), but I manually checked them on GeoSampa, and they didn’t seem to have any pools in them, so I just ignored them and kept going with the other 4177 images.

### Counting the pools
With the images downloaded, I could now run my pool detector on them. I used SAHI again, this time for sliced inference. The model would only see a single 1024 by 1024 tile at a time, and the predictions would be automatically merged into a single file per original image.

```
from sahi import AutoDetectionModel
import os
import json
from sahi.predict import get_sliced_prediction
from tqdm import tqdm

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="/mnt/nimbus/best.pt",
    image_size=1024,
    confidence_threshold=0.214,
    device="cuda",
    load_at_init=True,
)

folder_path = "/mnt/nimbus/images/originals"
jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
total = len(jpg_files)

for filename in tqdm(jpg_files, desc="Inference", colour='green'):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=1024,
            slice_width=1024,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=False,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.25,
            verbose=0,
        )

        coco_results = result.to_coco_annotations()
        output_file = f"/mnt/nimbus/output/{os.path.splitext(filename)[0]}.json"
        with open(output_file, "w") as f:
            json.dump(coco_results, f, indent=4)
```

Result:

```
raw_pool_count = 56,183
```

### Correction and processing
Now that I had the total raw pool count, I decided to apply a simple correction using the model’s validation metrics:

```
corrected_count = raw_pool_count * precision / recall
corrected_count = 56,183 * 0.86498 / 0.85774
corrected_count = 56,657
```

I also calculated the 95% confidence interval of the pool count using the bootstrap method:

```
95% CI: [54363, 59415]
```

To build a heat map of pool density in São Paulo, I would need to find the rough coordinates of every detected pool. I first downloaded a file from GeoSampa containing the coordinates of all the images, and then converted each detected pool’s local coordinates (in pixels, relative to the original image) to global coordinates (latitude and longitude):

```
import math
import json
import os
from tqdm import tqdm

# Configuration
annot_folder = '/mnt/nimbus/output'
output_json_path = '/mnt/nimbus/ortofotos.json'
result_json_path = '/mnt/nimbus/results.json'

def get_gsd(scale):
    """Get Ground Sample Distance based on scale."""
    if scale == "1:1000":
        return 0.1
    elif scale == "1:5000":
        return 0.2
    else:
        raise ValueError(f"Unknown scale: {scale}")

def distance(p1, p2):
    """Calculate Euclidean distance between two points [lat, lon]."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def find_corner_points(coords):
    """
    Find the 4 corner points from a polygon with any number of vertices.
    Returns dict with 'top_left', 'top_right', 'bottom_left', 'bottom_right'.
    Coordinates are [lat, lon].
    """
    # Remove closing point if present
    points = coords[:-1] if len(coords) > 1 and coords[0] == coords[-1] else coords
    
    # Remove duplicate points
    unique_points = []
    for p in points:
        if not unique_points or p != unique_points[-1]:
            unique_points.append(p)
    points = unique_points
    
    # Find bounding box - coords are [lat, lon]
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)
    
    # Define ideal corner positions [lat, lon]
    ideal_corners = {
        'top_left': [max_lat, min_lon],      # North-West
        'top_right': [max_lat, max_lon],     # North-East
        'bottom_left': [min_lat, min_lon],   # South-West
        'bottom_right': [min_lat, max_lon]   # South-East
    }
    
    # Find actual polygon points closest to each ideal corner
    corners = {}
    for corner_name, ideal_pos in ideal_corners.items():
        closest_point = min(points, key=lambda p: distance(p, ideal_pos))
        corners[corner_name] = closest_point
    
    return corners

def bilinear_interpolation(pixel_x, pixel_y, img_width, img_height, corners):
    """
    Convert pixel coordinates to geographic using bilinear interpolation.
    corners: dict with 'top_left', 'top_right', 'bottom_left', 'bottom_right'
    Each corner is [lat, lon]
    Returns: (lat, lon)
    """
    # Normalize pixel coordinates to [0, 1]
    norm_x = pixel_x / img_width
    norm_y = pixel_y / img_height
    
    # Get corner coordinates [lat, lon]
    tl = corners['top_left']
    tr = corners['top_right']
    bl = corners['bottom_left']
    br = corners['bottom_right']
    
    # Interpolate along top edge (y=0, at pixel row 0)
    top_lat = tl[0] + norm_x * (tr[0] - tl[0])
    top_lon = tl[1] + norm_x * (tr[1] - tl[1])
    
    # Interpolate along bottom edge (y=1, at pixel row height)
    bottom_lat = bl[0] + norm_x * (br[0] - bl[0])
    bottom_lon = bl[1] + norm_x * (br[1] - bl[1])
    
    # Interpolate vertically between top and bottom
    lat = top_lat + norm_y * (bottom_lat - top_lat)
    lon = top_lon + norm_y * (bottom_lon - top_lon)
    
    return lat, lon

def pixel_to_geographic(pixel_x, pixel_y, img_width, img_height, coords):
    """
    Convert pixel coordinates to geographic coordinates.
    Works for polygons with any number of vertices.
    Coordinates are [lat, lon].
    Returns: (lat, lon)
    """
    corners = find_corner_points(coords)
    return bilinear_interpolation(pixel_x, pixel_y, img_width, img_height, corners)

# Load output.json
with open(output_json_path, 'r') as f:
    image_metadata = json.load(f)

# Create a lookup dictionary
metadata_dict = {img['name']: img for img in image_metadata}

# Collect all processed objects
all_objects = []

# Process each annotation file
if os.path.exists(annot_folder):
    for filename in tqdm(os.listdir(annot_folder), desc="Processing", colour='green'):
        if not filename.endswith('.json'):
            continue
        
        # Extract name (without .json extension)
        name = filename[:-5]
        
        # Check if metadata exists
        if name not in metadata_dict:
            continue
        
        # Get image metadata
        img_meta = metadata_dict[name]
        img_width = img_meta['width']
        img_height = img_meta['height']
        coords = img_meta['coordinates']  # [lat, lon] format
        scale = img_meta['scale']
        gsd = get_gsd(scale)
        
        # Load annotation file
        annot_path = os.path.join(annot_folder, filename)
        with open(annot_path, 'r') as f:
            annotations = json.load(f)
        
        # Process each object
        for obj in annotations:
            bbox = obj['bbox']  # [x, y, width, height] in pixels
            pixel_x = bbox[0]
            pixel_y = bbox[1]
            pixel_width = bbox[2]
            pixel_height = bbox[3]
            
            # Convert top-left corner to geographic coordinates
            lat, lon = pixel_to_geographic(pixel_x, pixel_y, img_width, img_height, coords)
            
            # Convert dimensions to meters
            width_m = pixel_width * gsd
            height_m = pixel_height * gsd
            
            # Convert area to square meters
            area_m2 = obj['area'] * (gsd ** 2)
            
            # Create new object with [lat, lon] format to match input
            new_obj = {
                'bbox': [lat, lon, width_m, height_m],
                'score': obj['score'],
                'category_id': obj['category_id'],
                'category_name': obj['category_name'],
                'area': area_m2,
                'image_name': name
            }
            
            all_objects.append(new_obj)

# Save results
with open(result_json_path, 'w') as f:
    json.dump(all_objects, f, indent=4)

print(f"Processed {len(all_objects)} objects from {len(set(obj['image_name'] for obj in all_objects))} images")
print(f"Results saved to {result_json_path}")
```

The result was a single JSON file containing all the detected pools and their rough coordinates. Coordinate accuracy wasn’t amazing, but it was good enough to build a heat map.

The script above also converted the pool area from pixels to square meters, so I could calculate the total pool area.

```
Total pool area = 2,083,342 m² = 2.08 km²
```

### Building the website
With all the results in hand, I quickly built a simple website to present the final numbers and the heat map. I used Claude Sonnet 4.5 to build it.

I used [Folium](https://python-visualization.github.io/folium/latest/) to create an interactive heat map using the pools’ coordinates. I hosted the website on GitHub Pages.

I didn’t want to buy a domain for this project, but I also didn’t want to use the default GitHub Pages URL. Since I already had the [iglu.website](https://iglu.website) domain for another project, I decided to use [pools.iglu.website](https://pools.iglu.website) to host this new website.

## Conclusion
This project was fun! It was my first time working with computer vision, so I learned a lot. Though the end result doesn’t matter that much (who cares about the number of swimming pools in São Paulo?), I now have some new and very useful skills (and a great conversation starter).

There’s clearly still room for improvement here: the model seems to be overcounting pools and detecting some in the middle of lakes or forests. Maybe tuning the IoU value responsible for deduplication, or adding more negative samples of lakes and forests, could help.

All the scripts I used here were written (sometimes entirely) by AI (Grok, Claude, and ChatGPT).

Results are available at [pools.iglu.website](https://pools.iglu.website).
