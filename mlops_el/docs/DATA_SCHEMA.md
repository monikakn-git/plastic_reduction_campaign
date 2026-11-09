# Dataset rows and columns — brief explanation

The model uses photographs and bounding boxes. The following shapes describe the metadata arrays in the verified `annotations.json`, interpreted as tables; they are not the number of numeric features given to the detector.

| Logical table | Rows | Fields/columns | One row represents |
| --- | ---: | ---: | --- |
| `images` | 10,000 | 16 | One original photograph and its metadata |
| `annotations` | 60,826 | 9 | One labelled object in a photograph |

An image can contain several labelled objects, which is why annotation rows outnumber image rows. The average is about 6.08 annotations per image.

Important image fields include `id`, `file_name`, `width`, `height`, `dataset_id`, and `group_id`. Several other fields are source-tool bookkeeping. Source fields such as `num_annotations` and `annotated` are not treated as authoritative object counts; counts are computed from the actual annotation records.

Annotation fields are `id`, `image_id`, `category_id`, `area`, `bbox`, `iscrowd`, `isbbox`, `color`, and `metadata`. The `bbox` field contains four numbers: **left x, top y, width, height**. Expanding that nested field into four separate columns would make a flattened annotation table have 12 columns instead of 9.

There are **10 detection classes**. Source category IDs run from 1 to 10; the versioned class map converts them to YOLO's zero-based IDs from 0 to 9. Labels such as bottle and cup describe object types and do not certify plastic material.

Each RGB image itself has a pixel shape of **height × width × 3 colour channels**, with original dimensions varying across images. Baseline and YOLO11s training use nominal 640-pixel inputs; the final YOLO11m experiment uses nominal 960-pixel inputs with aspect-preserving resizing/padding. Stored originals retain their original resolution.

Exactly 10,000 distinct image hashes are in the final selection. Two duplicate-photo copies were replaced by unused annotated originals and preserved separately in the duplicate audit trail. Train/validation/test assignment keeps inferred source-video groups together.

DVC's total file count also includes manifests, annotations, and audit files. It should not be interpreted as the number of training images.
