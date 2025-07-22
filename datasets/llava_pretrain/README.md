# LLaVA Visual Instruct Pretrain Dataset

- dataset root dir: /capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain

```text
LLaVA-Pretrain/
├── 00001/ ... 00659/           # image subfolders (if unzipped)
├── blip_laion_cc_sbu_558k.json
├── blip_laion_cc_sbu_558k_meta.json
├── blip_laion_cc_sbu_558k_filtered.json   # filtered JSON (by resolution)
├── images.zip
└── README.md
```

## Dataset details

**Dataset type:**  
LLaVA Visual Instruct Pretrain LCS-558K is a subset of LAION/CC/SBU, filtered with a more balanced concept coverage distribution. Captions are also associated with [BLIP synthetic captions](https://github.com/salesforce/BLIP#pre-training-datasets-download) for reference.  
It is constructed for the pretraining stage for feature alignment in visual instruction tuning, aiming to build large multimodal models toward GPT-4 vision/language capability.

## Dataset Structure

- `blip_laion_cc_sbu_558k.json`: Contains the multimodal synthesized conversations from image-caption pairs, with randomly selected instructions (e.g., "Describe this image"). Raw CC-3M captions are used as default answers.
- `blip_laion_cc_sbu_558k_meta.json`: Metadata for each image, including file name, image URL, and BLIP synthetic caption.
- `images.zip`: All raw images of the filtered subset from LAION/CC/SBU. **Use of these images must comply with the LAION/CC/SBU license.**  
- **Filtered JSON**:  
  `blip_laion_cc_sbu_558k_filtered.json`  
  *Location*: `/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/blip_laion_cc_sbu_558k_filtered.json`  
  This file contains annotations for images after filtering by resolution.


## JSON Annotation Format

The main annotation file (`blip_laion_cc_sbu_558k_filtered.json`) is a list of dictionaries, each describing an image and its associated synthesized conversation.

**Each entry has the following fields:**

- `id`:  
  The unique identifier for the image, matching the filename (without extension).

- `image`:  
  Relative path to the image file, usually in the format `<subfolder>/<filename>.jpg`.  
  *Example: `"00453/004539375.jpg"`*

- `conversations`:  
  A list of dialog turns between a 'human' (instruction) and 'gpt' (response).
  Each element is a dictionary with:
  - `from`: `"human"` or `"gpt"`
  - `value`: The instruction or response text

**Example entry:**
```json
[
  {
    "id": "001819539",
    "image": "00181/001819539.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "Describe the image concisely.\n<image>"
      },
      {
        "from": "gpt",
        "value": "a pair of childrens boots in the shape of knitted shoes"
      }
    ]
  },
  {
    "id": "002838027",
    "image": "00283/002838027.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "Share a concise interpretation of the image provided.\n<image>"
      },
      {
        "from": "gpt",
        "value": "floral watercolor clip art rose flower border with pink"
      }
    ]
  },
// ...
]
```


## Modifications and Filtering

- **Resolution Filtering:**  
  To ensure consistent quality and manageable image size for training, images with either width or height greater than 1000 pixels were removed from the dataset.
- **Filtering results:**  
  - Number of images removed: **1827**
  - Number of images after filtering: **556,301**
  - Filtered annotation file:  
    `/capstor/store/cscs/swissai/infra01/vision-datasets/llava_pretrain/LLaVA-Pretrain/blip_laion_cc_sbu_558k_filtered.json`

## Image Resolution Statistics (Filtered)

- **Number of images:** 556,301
- **Width:** min=336, max=1000, mean=407.07, median=336.0
- **Height:** min=336, max=1000, mean=366.92, median=336.0
- **Percentiles:**  
  5th: 336x336  
  25th: 336x336  
  50th: 336x336  
  75th: 461x336  
  95th: 597x504
- **Aspect Ratio (W/H):** mean=1.15, median=1.00
- **Most common resolutions (top10):**  
  336x336: 171,011  
  448x336: 50,256  
  504x336: 28,409  
  597x336: 15,706  
  336x504: 14,937  
  336x448: 10,838  
  505x336: 6,504  
  423x336: 5,087  
  503x336: 4,825  
  336x420: 4,323  
- **Number of unique resolutions:** 1,313


