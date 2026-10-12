# FIT_TEACHER_CANONICAL_FEATURES.npz

Required arrays:

- `none`: shape `[N, 256]`
- `clean_kd`: shape `[N, 256]`
- `teacher`: shape `[N, 256]`

Rows must be paired representations of the same spatial P4 cell in the same image/view.
The file is for coordinate-map fitting only and must come from a pre-outcome TRAIN/FIT cohort.
It must not be selected using VAL outcomes or PRESERVED/LOST labels.

Companion metadata JSON must contain at minimum:

```json
{
  "split": "TRAIN",
  "val_used": false,
  "pres_lost_labels_used": false
}
```

The public package deliberately keeps this contract simple so a released artifact bundle can be regenerated independently from the model-specific extraction driver.
