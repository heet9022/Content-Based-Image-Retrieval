# Content-Based Image Retrieval

## Summary

This project implements various content-based image retrieval techniques such as chromatic matching and texture matching using C++ and the OpenCV library. The following techniques were experimented with:

### Baseline

A baseline was set using 9x9 pixels from the center of the images as features, with sum-of-squared-differences as the distance metric.

### 3D BGR Histogram Matching

A normalized 3D Histogram with 8 bins was created to capture the chromatic features of an image. The histogram intersection technique was used as the distance metric to find matches in the database.

### Multi-Histogram Matching

This technique uses a combination of two normalized 3D BGR histograms built over the upper and lower half of the image as features. The histogram intersection was used to measure the difference between the target image and the database.

### Color-Texture Histogram Matching

The Sobel filter was used to capture the texture features and combined with a 3D BGR histogram. The histogram intersection was used to find matches for the target image.

### HSV Chromatic Feature Extraction and Histogram Matching

A dataset of "orange_and_noranges" was created. A 3D BGR Histogram method was initially tried but resulted in poor accuracy due to changes in lighting. The 3D HSV Histogram model was then used and improved results were obtained, but the model was susceptible to noise. A Bilateral filter was applied to address the noise in the background, but the method was computationally expensive, so only the HSV histogram method was used.

## Extension

The Gabor filters with different theta values and the Histogram of Oriented Gradients method were implemented. A histogram of 8 bins was created, allowing the filter to recognize textures at different orientations. The Gabor filter was finally used.

## Key Takeaways

- Importance of understanding data types and storage in OpenCV
- Importance of code design before coding
- Better understanding of histograms

## Acknowledgment

The following websites were referred to learn about histograms:
- https://docs.opencv.org/3.4/index.html
- https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
- https://blog.actorsfit.com/a?ID=00450-3514e1f1-024f-40bd-ae01-e8070c74a990
- https://majeek.github.io/tutorials/scribbleSegmentation/
- https://linuxtut.com/en/ff4a07d73e14385b6922/
- https://en.wikipedia.org/wiki/HSL_and_HSV
