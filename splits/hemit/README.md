# HEMIT paper-test split

`hemit_test_no_overlap_no_empty_ids.txt` lists the 292 HEMIT test sample IDs
used for the paper evaluation. The list contains identifiers only; it does not
contain HEMIT image data.

The original 1024x1024 tiles were extracted from WSIs at a 512-pixel stride.
To remove overlap, we retained tiles whose `_patch_<row>_<column>` indices are
both even. The six IDs in `hemit_test_empty_excluded_ids.txt` were subsequently
excluded as empty tiles.

To regenerate the released selection from a file containing all original test
filenames, run:

```bash
python scripts/generate_hemit_test_split.py \
  --all-test-ids /path/to/all_hemit_test_ids.txt \
  --output /path/to/hemit_test_no_overlap_no_empty_ids.txt
```

The HEMIT dataset itself is governed by its original access and redistribution
terms; users must obtain it through the dataset provider.
