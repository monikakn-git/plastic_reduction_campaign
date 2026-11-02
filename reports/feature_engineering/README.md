# EcoPulse AI - Feature Engineering Results

## Overview
This folder documents the previously generated feature engineering
output used in the EcoPulse AI waste detection project.

## Output file
- File: engineered_features.csv
- Location: artifacts/feature_engineering/
- Source: Existing Google Drive engineered dataset
- Total records: 13,064

## Dataset split

| Split | Records |
|---|---:|
| Training | 9,798 |
| Validation | 1,960 |
| Testing | 1,306 |
| Total | 13,064 |

## Extracted features

- Image ID and filename
- Image width and height
- Image aspect ratio
- Object count
- Mean normalized bounding-box area
- Mean bounding-box aspect ratio
- Class-wise object counts

## Waste detection classes

1. Litter
2. Pile
3. Face mask
4. Trash bin
5. Plastic bag
6. Bottle
7. Cup
8. Rope
9. Sachet
10. Straw

## Data preservation

The existing engineered_features.csv was reused.
The original Google Drive file was not modified.
No additional feature extraction or model training was performed.

## Integrity
A SHA256 checksum is stored alongside the CSV file.
