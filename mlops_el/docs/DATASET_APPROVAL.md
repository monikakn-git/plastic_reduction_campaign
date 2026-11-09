# Proposed dataset approval record — 10,000-image Phase 1

| Field | Record |
| --- | --- |
| Dataset | pLitterStreet — Street Level Plastic Litter Detection Dataset |
| Source institution | Geoinformatics Center, Asian Institute of Technology |
| Primary repository | https://github.com/gicait/pLitter |
| Fixed dataset release | https://zenodo.org/records/8288500 |
| Paper | https://arxiv.org/abs/2401.14719 |
| Official source population | 13,064 images and 79,101 annotations in the 10-class annotation file |
| Project dataset | Exactly 10,000 selected original images; seed 42 |
| Licence | CC BY 4.0, declared in the official Zenodo API metadata |
| Academic approval | **Pending faculty approval** |
| Approved by/date | Not yet provided |

## Selection and integrity

The selection is deterministic and recorded in versioned annotations and image manifests. Source originals are extracted directly from the official ZIP using HTTP byte ranges. Each selected entry is validated against its published ZIP CRC32; its original bytes and SHA-256 are retained. The annotation archive is checked against the official MD5. The full image ZIP checksum is recorded as published, but is not claimed independently verified when only selected entries are downloaded.

No augmented images, resized copies, or repeated files are used to reach 10,000. The acquisition cannot be marked complete if any selected image fails validation or duplicate image hashes reduce the unique count below 10,000.

Two exact duplicate-photo copies were found in the initial selection. The lowest source image ID was retained for each identical-photo group, and unused annotated source images were selected in the same seeded candidate order as replacements. The final verified selection contains 10,000 unique image hashes and 60,826 annotations. `selection.json` records every selected ID and duplicate exclusion; excluded originals and all source annotation variants are preserved for review.

## Labels and relevance

All ten source classes are retained: litter, pile, face mask, trash bin, plastic bag, bottle, cup, rope, sachet, and straw. The street-litter setting supports later location analysis and targeted campaigns. Generic bottle/cup/litter labels do not certify an object's material; the application must not infer that every detected object is plastic.

Source-video groups are inferred from source dataset IDs and filename stems. Groups remain in one split to reduce adjacent-frame leakage. Actual split counts and class coverage must be reviewed after acquisition; the intended proportions are approximately 70/15/15.

## Attribution and limitations

Cite Mandhati et al., pLitterStreet: Street Level Plastic Litter Detection and Mapping (2024), arXiv:2401.14719, and the dataset release DOI 10.5281/zenodo.8288500. Retain the authors' attribution and CC BY 4.0 notice.

Images represent the source survey regions, not this college campus. Public-dataset evaluation does not establish campus accuracy or campaign effectiveness. Strong class imbalance, small objects, source-domain differences, and inferred sequence grouping must be documented.

## Faculty decision

Pending. Record the accepted source, 10,000-image selection, class scheme, intended use, approver, date, and any conditions when the decision is provided.
