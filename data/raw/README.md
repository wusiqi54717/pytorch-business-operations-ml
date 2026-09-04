# Raw Data

Place the raw CSV files used by the scripts in this folder.

The main classification workflow expects:

```text
online_shoppers_intention.csv
```

This file is intentionally not committed here through the current upload flow because it is a relatively large raw dataset. The local project already contains it, and the preprocessing script will generate the processed `.npz` file from it.

The second local CSV, `Wholesale customers data.csv`, is not used by the current classification neural network workflow. It is kept locally for possible future regression or segmentation exploration.
