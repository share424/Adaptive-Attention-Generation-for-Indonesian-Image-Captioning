# Adaptive Attention Generation for Indonesian Image Captioning
this code is implementation of [Adaptive Attention Generation for Indonesian Image Captioning](https://drive.google.com/file/d/1GZXQFF5RKpElZWKL9yK-2Wynh0edwJC5/view) paper 

![image](diagram/architecture.svg)

# Changelog
- 2020-01-15: Initial commit
- 2024-03-10: Refactor the code, add support to pytorch 2.x, and add more documentation

# Evaluation Results
| Model | Dataset | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | Link |
|-------|--------|--------|--------|--------|--------|---------|-------|-------|-------|
| ResNet101 & LSTM | Flickr30K | 0.695 | 0.539 | 0.403 | 0.299 | 0.256 | 0.544 | 0.895 | [Download](https://github.com/share424/Adaptive-Attention-Generation-for-Indonesian-Image-Captioning/releases/download/Weights/weights-resnet101.zip) |
| ResNet101 & LSTM | COCO2014 val | 0.667 | 0.497 | 0.358 | 0.257 | 0.245 | 0.509 | 0.967 | [Download](https://github.com/share424/Adaptive-Attention-Generation-for-Indonesian-Image-Captioning/releases/download/Weights/weights-resnet101.zip) |

**More models comming soon**

# Requirements
- Python 3.10
- Java 1.8 (Required by the evaluation module, if you want to inference only, you can skip this)

# Installation

## Google Colab
Follow this [example](https://colab.research.google.com/github/share424/Adaptive-Attention-Generation-for-Indonesian-Image-Captioning/blob/master/notebooks/example.ipynb).

## With GPU CUDA
1. Makesure you have installed [CUDA and cuDNN](https://medium.com/geekculture/install-cuda-and-cudnn-on-windows-linux-52d1501a8805)
2. Install pytorch 2.2.x with CUDA support, you can follow this [link](https://pytorch.org/get-started/previous-versions/#v220), or you can just install the latest version
3. Clone this repo or download the zip file
    ```bash
    git clone https://github.com/share424/Adaptive-Attention-Generation-for-Indonesian-Image-Captioning.git
    ```
4. Install the requirements
    ```bash
    python setup.py install
    ```

## Without GPU
1. Clone this repo or download the zip file
    ```bash
    git clone https://github.com/share424/Adaptive-Attention-Generation-for-Indonesian-Image-Captioning.git
    ```
2. Install the requirements
    ```bash
    python setup.py install
    ```

# Predict a single image
We already provide a script to predict a single image, here's how to use it
```bash
Usage:
python scripts/predict_single.py <config> [--ckpt <checkpoint>] [--image <image>] [--search-strategy <strategy>] [--beam-size <size>] [--visualize] [--device <device>]

positional arguments:
  config                Path to the configuration file

options:
  -h, --help            show this help message and exit
  --ckpt CKPT           Path to the checkpoint file
  --image IMAGE         Path to the image file
  --search-strategy {greedy_search,beam_search}, -s {greedy_search,beam_search}
                        Search strategy to use
  --beam-size BEAM_SIZE
                        Beam size
  --visualize           Visualize the attention
  --device DEVICE       Device to use
```

Here is the example
1. Download and unzip the pretrained models from the [Evaluation Results table](#evaluation-results)
2. Run the following command
    ```bash
    python scripts/predict_single.py resnet101-lstm.yaml --ckpt BEST_epoch_18_resnet101.pth -image images/sample-2.jpg
    ```
    output
    ```bash
    ['kucing duduk di atas meja kayu']
    ```
If you found an error like this
```
FileNotFoundError: [Errno 2] No such file or directory: 'wordmap.json'
```
update the `resnet101-lstm.yaml` and edit the `tokenizer.wordmap` section to the absolute path of the `wordmap.json` file

# Train the model
Before we start, you need to understand our training pipeline

## Training Pipeline
1. Freeze encoder, and train decoder only for N - k epochs
2. Unfreeze encoder, and train both encoder and decoder for k epochs

## Dataset Preparation
1. Create a `config.yaml` files, you can use the `config/resnet101-lstm.yaml` as a reference
2. Download the image dataset, you can use the [Flickr30K](https://www.kaggle.com/hsankesara/flickr-image-dataset) or [COCO2014](https://cocodataset.org/#download) dataset
3. Download the translated Indonesia dataset [here](https://github.com/share424/Adaptive-Attention-Generation-for-Indonesian-Image-Captioning/releases/tag/Dataset)
4. Adjust the `config.yaml` file to the dataset path
   ```yaml
   data:
    train:
        - annotation: dataset/coco2014_indo_train.json
        image_dir: dataset/coco2014/train2014
        - annotation: dataset/flickr30k_indo_train.json
        image_dir: dataset/flickr30k_images
    validation:
        - annotation: dataset/flickr30k_indo_val.json
        image_dir: dataset/flickr30k_images
    test:
        - annotation: dataset/flickr30k_indo_test.json
          image_dir: dataset/flickr30k_images
        - annotation: dataset/coco2014_indo_val.json
          image_dir: dataset/coco2014/val2014
   ```

## Create Wordmap
Wordmap is a dictionary that maps the word to the index, you can create the wordmap by running the following command
```bash
python scripts/create_wordmap.py dataset/coco2014_indo_train.json dataset/flickr30k_indo_train.json --output wordmap.json
```
**note: use the train dataset to avoid the models see the test-set**

You can add more dataset to the `create_wordmap.py` script, here's the usage
```
Usage:
    python scripts/create_wordmap.py <annotations>... [--output <output>]

Arguments:
    annotations (List[str]): List of paths to COCO annotations.
    output (str): Path to the output wordmap JSON file.

Example:
    python scripts/create_wordmap.py annotations.json --output wordmap.json
```

**Note: Don't forget to add the wordmap path to the `config.yaml` file**
```yaml
tokenizer:
  wordmap: wordmap.json
  max_length: 50
```

## Training Configuration
1. Edit the `config.yaml` file to adjust the training configuration
```yaml
training:
  epochs: 20
  finetune_epochs: 5 # 5 last epoch is finetune encoder
  finetune_n_layer: 5 # num of last layer that will be finetune
  grad_clip: 10.0 # gradient clip value to avoid exploding gradient
  encoder_optimizer:
    name: Adam
    config:
      betas: [0.9, 0.999]
      lr: 0.0001
    lr_scheduler:
      name: ReduceLROnPlateau
      config:
        mode: max
        patience: 3
        verbose: True
        factor: 0.1

  decoder_optimizer:
    name: Adam
    config:
      betas: [0.8, 0.999]
      lr: 0.0005
    lr_scheduler:
      name: ReduceLROnPlateau
      config:
        mode: max
        patience: 3
        verbose: True
        factor: 0.1
  
  loss:
    name: CrossEntropyLoss
    config:
      reduction: mean

  checkpoint_dir: checkpoints
  track_metric: BLEU-1 # used by lr_scheduler and early stopping
  # available: [BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE_L, CIDEr, loss, top5_accuracy]
  early_stop_n_epoch: 5
```

## Training script
We already provide a script to train the model, here's how to use it
```bash
Usage:
    python scripts/train.py <config> [--device <device>] [--ckpt <checkpoint>]

Arguments:
    <config>                Path to the configuration file.
    --device <device>       Device to use. Default is 'cuda'.
    --ckpt <checkpoint>     Path to the checkpoint file. Optional.

Example:
    python scripts/train.py config.yaml --device cuda --ckpt checkpoint.pth

```

Here is the example
```bash
 python scripts/train.py config.yaml
```

the checkpoints will be saved in the `checkpoints` directory based on the `config.yaml` file

# Evaluation
We already provide a script to evaluate the model, here's how to use it
```bash
Usage:
    python scripts/eval.py <config> [--split <split>] [--device <device>] [--ckpt <checkpoint>]

Arguments:
    config (str): Path to the configuration file.
    
Options:
    --split (str): Split to evaluate. Default is 'test'.
    --device (str): Device to use. Default is 'cuda'.
    --ckpt (str): Path to the checkpoint file.

Example:
    python scripts/eval.py config.yaml --split val --device cuda:0 --ckpt model.ckpt
```

Here is the example
```bash
python scripts/eval.py config.yaml --split test --device cuda:0 --ckpt model.ckpt
```

# Acknowledgement
- [Adaptive Attention Generation for Image Captioning](https://drive.google.com/file/d/1GZXQFF5RKpElZWKL9yK-2Wynh0edwJC5/view)
- [knowing-when-to-look-adaptive-attention](https://github.com/fawazsammani/knowing-when-to-look-adaptive-attention)

# Citation
```bibtex
@INPROCEEDINGS{Sury2006:Adaptive,
    AUTHOR="Made {Surya Mahadi} and Anditya Arifianto and Kurniawan {Nur Ramadhani}",
    TITLE="Adaptive Attention Generation for Indonesian Image Captioning",
    BOOKTITLE="2020 8th International Conference on Information and Communication
    Technology (ICoICT) (ICoICT 2020)",
    ADDRESS="Yogyakarta, Indonesia",
    DAYS=23,
    MONTH=jun,
    YEAR=2020,
}
```

# License
This project is licensed under the MIT License.