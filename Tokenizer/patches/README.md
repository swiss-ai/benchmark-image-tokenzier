# Emu3.5 Spatial Indices Patch

This patch modifies Emu3.5 to return spatial indices [B, H, W] instead of flattened indices, which is required for the Emu3_5_IBQ tokenizer.

## Apply the patch

```bash
cd Tokenizer/submodules/Emu3.5
git apply ../../patches/emu35_spatial_indices.patch
```