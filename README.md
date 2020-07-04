# Multi-Temporal Recurrent Neural Networks For Progressive Non-Uniform Single Image Deblurring With Incremental Temporal Training (Accepted for ECCV 2020 Spotlight)
[paper](https://arxiv.org/abs/1911.07410)
We provide PyTorch implementations for MTRNN.
Note: The current software works well with PyTorch 0.41+.
# Prerequisites
Linux or macOS
Python 3
CPU or NVIDIA GPU + CUDA CuDNN

# Abstract
Blind non-uniform image deblurring for severe blurs induced by large motions is still challenging. Multi-scale (MS) approach has been widely used for deblurring that sequentially recovers the downsampled original image in low spatial scale first and then further restores in high spatial scale using the result(s) from lower spatial scale(s). Here, we investigate a novel alternative approach to MS, called multi-temporal (MT), for non-uniform single image deblurring by exploiting time-resolved deblurring dataset from high-speed cameras. MT approach models severe blurs as a series of small blurs so that it deblurs small amount of blurs in
the original spatial scale progressively instead of restoring the images in different spatial scales. To realize MT approach, we propose progressive deblurring over iterations and incremental temporal training with temporally augmented training data. Our MT approach, that can be seen as a form of curriculum learning in a wide sense, allows a number of stateof-the-art MS based deblurring methods to yield improved performances without using MS approach. We also proposed a MT recurrent neural network with current feature maps that outperformed state-of-the-art deblurring methods with the smallest number of parameters.

# Test

python3 test.py

# Train
Comming Soon
