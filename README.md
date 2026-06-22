# SRCNNQuantized
This project aims to compare the effects of different quantization methods for SRCNN super resolution model. deployment on MAX78000FTHR board.  The project was created in the scope of "EE634: Digital Image Processing" offered in METU on 2026 Spring.

## Introduction
In this project, SRCNN [1] was used to test the performance degradation caused by quantization errors observed in Post Training Quantization (PTQ) and Quantization Aware Training (QAT). 

## Dependencies

This project started while PyTorch was migrating torch.ao to a separate library: torchao. However, because the migration was only halfway done, major CNN quantization capabilities were left in the soon to be deprecated torch.ao which is used in this project. As long as the module versions in dependencies.txt is used, the project should run normally.

## Dataset

DIV2K [2] dataset was used to train SRCNN and tune the QAT process. 800 high resolution (HR) training images were augmented with random vertical and horizontal flips, random image clipping (512x512), and random 90° rotations. Both validation and test sets were augmented during training.

66 validation images were used from the dataset at every epoch to validate the model. SSIM and MSE metrics were used during validation and testing steps.

34 images were used from the dataset at the end of the training to test the performances, 4 of which is displayed in the results section with their respective SSIM values.

Retrospectively, it could be better to not augment the validation data as the augmentation made the metric curves less stable due to changing statistics.

## Model

![SRCNN](img/SRCNN.svg)

A Standard SRCNN model was selected to demonstrate the effects of PTQ and QAT. The model has a 64 channel 9x9x1 feature extraction layer, a 32 channel 5x5x64 non-linear mapping layer, and a 5x5x32 reconstruction layer.

The model was trained with the augmented DIV2K dataset for 50 epochs. The training epoch count might not be optimal as it was purposely selected lower to decrease training time. As the project is more focused on the effects of quantization, 50 epochs were thought to be enough.

### Quantization

Two different quantization methods were used to compress SRCNN: PTQ and QAT. It could be said that PTQ is the fastest and simplest way to quantize a model. However, different configurations exist which makes the PTQ pipeline a little less straightforward than it initially appears.

Two main options are available for PTQ in torch.ao: dynamic and static quantization. The main difference is dynamic quantization handles the quantization scaling online with a tradeoff of slight overhead while static quantization requires calibration inferences while observers are planted in the model

#### Observers

#### FX graph

## Workflow

![ProjectDiagram](img/ProjectDiagram.drawio.svg)

## Results

### SSIM

### Test Images


## Future Work / Shortcomings

One of the main intended goals of the project was to deploy the quantized models to MAX78000FTHR as well as trying out the ai8x training and synthesis frameworks. However, these could not be implemented in time, and therefore, are listed as future work.

## Directories

maxim: Not added as it does not contain extra work other than what is provided in MaximSDK.

img: Markdown images.

img/results: Result images and plots

torch: PyTorch related codes. All the code is written as jupyter notebooks for ease of debugging however they might be converted to regular .py files if deemed necessary.

## References

[1] C. Dong, C. C. Loy, K. He, ve X. Tang, “Image Super-Resolution Using Deep Convolutional Networks”, 31 Temmuz 2015, arXiv: arXiv:1501.00092. doi: 10.48550/arXiv.1501.00092.

[2] https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images
