# Example inputs

Two synthetic training-style photos from the validation split, exactly as stored
in the dataset (448×576 JPEG). Use them to check that inference works:

```bash
python training/inference.py --ckpt weights/dewarpnet_layout_lambda0.2.pt \
    --image examples/synthetic_cylinder_PMC6091653_00001.jpg --out fixed.png --device cpu
```

| file | warp profile | source page |
|---|---|---|
| `synthetic_cylinder_PMC6091653_00001.jpg` | cylinder | PubLayNet `PMC6091653_00001` |
| `synthetic_combined_PMC4041224_00000.jpg` | combined (cylinder + perspective + creases) | PubLayNet `PMC4041224_00000` |

These are derived from PubLayNet (Zhong et al., ICDAR 2019), which is licensed
under CDLA-Permissive-1.0 (<https://cdla.io/permissive-1-0/>). The underlying
pages come from the PubMed Central Open Access subset.
