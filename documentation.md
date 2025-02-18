# Leaffliction

## Summary

## Transformation

### Visualize Colorspace

The visualization tool converts the color image into HSV and LAB colorspaces and displays the grayscale channels in a matrix so that they can be visualized simultaneously. The idea is to select a channel that maximizes the difference between the plant and the background pixels. 

#### HSV (Hue, Saturation, Value)

- **Hue**: Represents the color type (e.g., red, green, blue). It is an angle from 0 to 360 degrees.
- **Saturation**: Represents the intensity or purity of the color. It ranges from 0 (gray) to 100% (full color).
- **Value**: Represents the brightness of the color. It ranges from 0 (black) to 100% (full brightness).

The HSV colorspace is often used in image processing because it separates the color information (hue) from the intensity information (value), making it easier to perform tasks like color-based segmentation.

#### LAB

- **L**: Represents the lightness of the color, ranging from 0 (black) to 100 (white).
- **A**: Represents the color component from green to red. Negative values indicate green, and positive values indicate red.
- **B**: Represents the color component from blue to yellow. Negative values indicate blue, and positive values indicate yellow.

The LAB colorspace is designed to be perceptually uniform, meaning that the same amount of numerical change in these values corresponds to roughly the same amount of visually perceived change. This makes it useful for tasks like color correction and color-based segmentation.

## Training

### Model

