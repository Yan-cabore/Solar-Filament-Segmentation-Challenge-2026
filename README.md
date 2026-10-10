# Filament Detection Strategy (Progress Notes)

## Goal
Segment solar filaments (dark, thin, elongated structures) in GONG H-Alpha grayscale images for the Kaggle Solar Filament Segmentation Challenge 2026. The work is split into stages, starting with building an image where the filaments are enhanced.

## Current status
**Stage 1 (enhancement) is done:** an image with enhanced filaments has been built and is ready to work with. Binarization and mask cleanup (including any standalone closing) have **not** been applied yet.

## Why the first attempts fell short
- **Sobel** (gradient magnitude) and **Laplacian** (second derivative) respond to the *edges* of a structure, so a filament produces two separate responses (one per margin) instead of one coherent object.
- Both are dominated by the **solar limb**, a strong circular edge that hides the filaments.
- The Laplacian output is signed, so zero shows up as mid-gray, which makes the comparison with Sobel visually misleading unless the absolute value is used.
- Sharpening followed by Gaussian blur is a composition of linear filters, equivalent to a single kernel (roughly a Difference of Gaussians), so it adds little.

## Chosen approach: Black-hat (bottom-hat) transform
Filaments are **dark, thin structures on a brighter background**, which is exactly what black-hat isolates.

```
black-hat(f) = closing(f) - f
```

- Internally, the closing fills dark structures narrower than the structuring element (SE), estimating the bright local background. Subtracting the original leaves only those thin dark structures.
- It also suppresses slow brightness variations (e.g. limb darkening) without extra steps.
- It works directly on grayscale images (grayscale morphology uses local min/max).

## Pipeline so far (enhancement only)
1. **Read** the image in grayscale.
2. **Gaussian blur** (light) to reduce fine granulation.
3. **Median blur** (3 or 5) to preserve and slightly enhance edge detail.
4. **Black-hat** with an elliptical SE (21x21).

## Images link
- To test this pipeline, you'll need to first install the images provided by the organizators. I pretend to upload all the files to a google Drive in the future, since downloading the images will require the creation of a keggle account.
- Link: [https://www.kaggle.com/competitions/filament-segmentation-2026]

```python
gaussian = cv.GaussianBlur(src, ksize=(3, 3), sigmaX=3)
median_3 = cv.medianBlur(gaussian, ksize=3)
median_5 = cv.medianBlur(gaussian, ksize=5)

ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21))
bhat_3 = cv.morphologyEx(median_3, cv.MORPH_BLACKHAT, ellipse)
bhat_5 = cv.morphologyEx(median_5, cv.MORPH_BLACKHAT, ellipse)
```

## Observations
- The **median filter enhances some edge details**, at the cost of slightly amplifying background noise. Comparing `median_3` and `median_5` is a trade-off between detail and noise.

## Next steps
- Binarize the enhanced image (threshold or hysteresis).
- Clean the binary mask with small morphology (opening or `remove_small_objects`), and optionally apply a closing to reconnect broken filaments.
