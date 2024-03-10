from typing import Tuple, List

from pydantic import BaseModel, Field
from yaml import safe_load


class EncoderConfig(BaseModel):
    name: str = Field(str, description="Name of the encoder")
    weights: str = Field(str, description="Path to the weights")
    freeze: bool = Field(False, description="Freeze the weights")
    config: dict = Field(dict, description="Configuration of the encoder")


class DecoderConfig(BaseModel):
    name: str = Field(str, description="Name of the decoder")
    weights: str = Field(str, description="Path to the weights")
    freeze: bool = Field(False, description="Freeze the weights")
    config: dict = Field(dict, description="Configuration of the decoder")


class ArchitectureConfig(BaseModel):
    encoder: EncoderConfig = Field(EncoderConfig, description="Encoder configuration")
    decoder: DecoderConfig = Field(DecoderConfig, description="Decoder configuration")


class CocoConfig(BaseModel):
    annotation: str = Field(str, description="Path to the annotation file")
    image_dir: str = Field(str, description="Path to the image directory")


class TokenizerConfig(BaseModel):
    wordmap: str = Field(str, description="Path to the wordmap file")
    max_length: int = Field(int, description="Maximum length of the caption")


class DataConfig(BaseModel):
    train: List[CocoConfig] = Field(CocoConfig, description="Path to the training data")
    validation: List[CocoConfig] = Field(CocoConfig, description="Path to the validation data")
    test: List[CocoConfig] = Field(CocoConfig, description="Path to the test data")
    batch_size: int = Field(int, description="Batch size")
    num_workers: int = Field(int, description="Number of workers", strict=True)


class LRSchedulerConfig(BaseModel):
    name: str = Field(str, description="Name of the learning rate scheduler")
    config: dict = Field(dict, description="Configuration of the learning rate scheduler")


class OptimizerConfig(BaseModel):
    name: str = Field(str, description="Name of the optimizer")
    config: dict = Field(dict, description="Configuration of the optimizer")
    lr_scheduler: LRSchedulerConfig = Field(LRSchedulerConfig, description="Learning rate scheduler configuration")


class LossConfig(BaseModel):
    name: str = Field(str, description="Name of the loss function")
    config: dict = Field(dict, description="Configuration of the loss function")


class TrainConfig(BaseModel):
    epochs: int = Field(int, description="Number of epochs")
    finetune_epochs: int = Field(int, description="Number of epochs for finetuning")
    finetune_n_layer: int = Field(int, description="Number of layers to finetune")
    grad_clip: float = Field(float, description="Gradient clipping value")
    encoder_optimizer: OptimizerConfig = Field(OptimizerConfig, description="Encoder optimizer configuration")
    decoder_optimizer: OptimizerConfig = Field(OptimizerConfig, description="Decoder optimizer configuration")
    loss: LossConfig = Field(LossConfig, description="Loss function configuration")
    checkpoint_dir: str = Field(str, description="Path to the checkpoint directory")
    track_metric: str = Field(str, description="Metric to track for early stopping")
    early_stop_n_epoch: int = Field(int, description="Number of epochs to wait for improvement")
    

class PreprocessConfig(BaseModel):
    image_size: Tuple[int, int] = Field((224, 224), description="Size of the image")
    center_crop_size: Tuple[int, int] = Field((224, 224), description="Size of the center crop")
    normalize_mean: Tuple[float, float, float] = Field((0.485, 0.456, 0.406), description="Mean of the normalization")
    normalize_std: Tuple[float, float, float] = Field((0.229, 0.224, 0.225), description="Standard deviation of the normalization")


class Config(BaseModel):
    architecture: ArchitectureConfig = Field(ArchitectureConfig, description="Architecture configuration")
    tokenizer: TokenizerConfig = Field(TokenizerConfig, description="Tokenizer configuration")
    data: DataConfig = Field(DataConfig, description="Data configuration")
    training: TrainConfig = Field(TrainConfig, description="Training configuration")
    preprocess: PreprocessConfig = Field(PreprocessConfig, description="Preprocessing configuration")

    @staticmethod
    def from_yaml(file_path: str) -> "Config":
        # TODO: somehow `model_validate` didn't validate empty fields
        with open(file_path, "r") as file:
            config = safe_load(file)
            return Config.model_validate(config)
            
            