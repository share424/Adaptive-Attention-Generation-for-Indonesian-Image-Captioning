
from typing import Union, List, Tuple, NamedTuple, Optional
import os

import torch
from torch import Tensor, nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

from image_captioning.models.model_registry import get_model
from image_captioning.config import Config, OptimizerConfig
from image_captioning.tokenizer import Tokenizer, SpecialToken
from image_captioning.visualize import visualize_attention
from image_captioning.dataset import COCOTextDataset, COCOImageDataset
from image_captioning.metrics import get_metric
from image_captioning.utils.torch_utils import optimizer_to, scheduler_to, set_requires_grad


class CheckpointData(NamedTuple):
    epoch: int
    best_score: float
    n_epoch_no_improvement: int
    is_finetune: bool


class AdaptiveAttentionImageCaptioning:
    def __init__(self, config: Config, **kwargs):
        super().__init__()
        self.config = config
        self.tokenizer = Tokenizer.from_config(config.tokenizer)

        self.encoder: nn.Module = get_model(
            config.architecture.encoder.name
        )(**config.architecture.encoder.config, **kwargs)
        self.decoder: nn.Module = get_model(
            config.architecture.decoder.name
        )(
            **config.architecture.decoder.config,
            start_token=self.tokenizer.get_index(SpecialToken.START),
            end_token=self.tokenizer.get_index(SpecialToken.END),
            vocab_size=self.tokenizer.vocab_size(),
            **kwargs
        )
        
        if config.architecture.encoder.weights:
            self._load_encoder_weights(
                config.architecture.encoder.weights
            )

        if config.architecture.decoder.weights:
            self._load_decoder_weights(
                config.architecture.decoder.weights
            )

    def _load_encoder_weights(self, encoder_weights: str):
        """
        Loads the weights of the encoder from a given file.

        Args:
            encoder_weights (str): The file path of the encoder weights.

        Returns:
            None
        """
        self.encoder.load_state_dict(torch.load(encoder_weights))

    def _load_decoder_weights(self, decoder_weights: str):
        """
        Loads the weights of the decoder model from a given file.

        Args:
            decoder_weights (str): The path to the file containing the decoder weights.

        Returns:
            None
        """
        self.decoder.load_state_dict(torch.load(decoder_weights))

    def load_checkpoint(self, checkpoint: str):
            """
            Loads a checkpoint file and updates the encoder and decoder models with the saved state.

            Args:
                checkpoint (str): The path to the checkpoint file.

            Returns:
                None
            """
            ckpt = torch.load(checkpoint)
            self.encoder.load_state_dict(ckpt["encoder"])
            self.decoder.load_state_dict(ckpt["decoder"])

    @staticmethod
    def from_yaml(path: str, **kwargs) -> "AdaptiveAttentionImageCaptioning":
        """
        Load the configuration from a YAML file and create an instance of AdaptiveAttentionImageCaptioning.

        Args:
            path (str): The path to the YAML file containing the configuration.
            **kwargs: Additional keyword arguments to be passed to the AdaptiveAttentionImageCaptioning constructor.

        Returns:
            AdaptiveAttentionImageCaptioning: An instance of the AdaptiveAttentionImageCaptioning class.

        """
        config = Config.from_yaml(path)
        return AdaptiveAttentionImageCaptioning(config, **kwargs)

    def _load_image(self, image: Union[Tensor, Image.Image, np.ndarray, str]) -> Image.Image:
        """
        Load an image from various input types.

        Args:
            image (Union[Tensor, Image.Image, np.ndarray, str]): The input image to load.

        Returns:
            Image.Image: The loaded image.

        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Tensor):
            image = Image.fromarray(image.numpy())

        return image
    
    @staticmethod
    def get_transform(config: Config) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(tuple(config.preprocess.image_size)),
            transforms.CenterCrop(tuple(config.preprocess.center_crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.preprocess.normalize_mean,
                std=config.preprocess.normalize_std
            )
        ])
    
    def _preprocess(self, images: List[Image.Image]) -> Tensor:
        """
        Preprocesses a list of images.

        Args:
            images (List[Image.Image]): A list of PIL images.

        Returns:
            Tensor: A tensor containing the preprocessed images.
        """
        transform = self.get_transform(self.config)
        return torch.stack([transform(image) for image in images])

    def predict_single(
        self, 
        image: Union[Tensor, Image.Image, np.ndarray, str],
        return_raw=False,
        keep_special_token=False,
        return_visualize=False,
        device: str = "cpu",
        **kwargs
    ):
        """
        Generates a caption for a single image.

        Args:
            image (Union[Tensor, Image.Image, np.ndarray, str]): The input image.
            return_raw (bool, optional): Whether to return the raw prediction, attention weights, and gating weights. Defaults to False.
            keep_special_token (bool, optional): Whether to keep special tokens in the generated captions. Defaults to False.
            return_visualize (bool, optional): Whether to return visualizations of attention weights. Defaults to False.
            device (str, optional): The device to run the model on. Defaults to "cpu".
            **kwargs: Additional keyword arguments to be passed to the decoder.

        Returns:
            - If return_raw is True, returns a tuple containing the prediction, attention weights, and gating weights.
            - If return_visualize is True, returns a tuple containing the generated sentences and a list of visualizations of attention weights.
            - Otherwise, returns the generated sentences.
        """
        self.encoder.eval()
        self.decoder.eval()
        self.encoder.to(device)
        self.decoder.to(device)

        inputs = self._preprocess([self._load_image(image)])
        inputs = inputs.to(device)

        image_features = self.encoder(inputs)
        prediction, alpha, beta = self.decoder(*image_features, **kwargs)

        if return_raw:
            return prediction, alpha, beta
        
        if return_visualize and not keep_special_token:
            raise ValueError("Must keep special tokens to visualize attention weights.")

        sentences = self.tokenizer.detokenize(prediction[0], keep_special_token=keep_special_token)

        if return_visualize:
            images = []

            for i, sentence in enumerate(sentences):
                visualized_image = visualize_attention(
                    image, 
                    sentence.split(" "), 
                    np.array(alpha[0][i]), 
                    np.array(beta[0][i])
                )
                images.append(visualized_image)

            return sentences, images
        
        return sentences
        
    def predict_batch(
        self, 
        images: List[Union[Tensor, Image.Image, np.ndarray, str]],
        keep_special_token=False,
        skip_preprocess=False,
        device: str = "cpu",
        **kwargs
    ):
        """
        Generate image captions for a batch of images.

        Args:
            images (List[Union[Tensor, Image.Image, np.ndarray, str]]): A list of images to generate captions for.
            keep_special_token (bool, optional): Whether to keep special tokens in the generated captions. Defaults to False.
            skip_preprocess (bool, optional): Whether to skip the preprocessing step. Defaults to False.
            device (str, optional): The device to run the model on. Defaults to "cpu".
            **kwargs: Additional keyword arguments to be passed to the decoder.

        Returns:
            List[str]: A list of generated image captions.
        """
        self.encoder.eval()
        self.decoder.eval()
        self.encoder.to(device)
        self.decoder.to(device)

        if not skip_preprocess:
            inputs = self._preprocess([self._load_image(image) for image in images])
        else:
            inputs = images

        inputs = inputs.to(device)

        image_features = self.encoder(inputs)
        predictions, _, _ = self.decoder(*image_features, **kwargs)
    
        sentences = []
        for prediction in predictions:
            sentence = self.tokenizer.detokenize(prediction, keep_special_token=keep_special_token)[0]
            sentences.append(sentence)

        return sentences
    
    def evaluate(self, device: str = "cpu", split: str = "test", **kwargs):
        import tempfile
        import json
        import os

        from pycocotools.coco import COCO
        from .coco_eval_cap import COCOEvalCap
            
        dataset = COCOImageDataset(
            configs=getattr(self.config.data, split), 
            tokenizer=self.tokenizer,
            return_keys=["image", "image_id"],
            transform=self.get_transform(self.config),
            process_captions=False,
        )
        data_loader = DataLoader(
            dataset, 
            batch_size=self.config.data.batch_size, 
            num_workers=self.config.data.num_workers
        )

        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.eval()
        self.decoder.eval()

        coco_predictions = []

        for images, image_ids in tqdm(data_loader, desc=f"Evaluating {split} set"):
            images = images.to(device)
            predictions = self.predict_batch(
                images, 
                ignore_special_token=True, 
                skip_preprocess=True,
                device=device,
                **kwargs
            )
            for image_id, prediction in zip(image_ids, predictions):
                coco_predictions.append({
                    "image_id": int(image_id.item()),
                    "caption": prediction
                })
            

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            json.dump(coco_predictions, f)
            coco_result_path = f.name

        final_metrics = {}

        for dataset in getattr(self.config.data, split):
            coco = COCO(dataset.annotation)
            len_data = len(coco.getImgIds())
            coco_result = coco.loadRes(coco_result_path)
            coco_eval = COCOEvalCap(coco, coco_result)
            metrics = coco_eval.evaluate()
            
            for k, v in metrics.items():
                if k not in final_metrics:
                    final_metrics[k] = []

                final_metrics[k].append((v, len_data))

        os.remove(coco_result_path)

        for k, v in final_metrics.items():
            final_metrics[k] = sum([m * n for m, n in v]) / sum([n for _, n in v])

        return final_metrics

    def _calculate_accuracy(self, predictions: Tensor, targets: Tensor, k: int = 5) -> float:
        batch_size = targets.size(0)
        # Return the indices of the top-k elements along the first dimension (along every row of a 2D Tensor), sorted
        _, ind = predictions.topk(k, 1, True, True)    
        # The target tensor is the same for each of the top-k predictions (words). Therefore, we need to expand it to  
        # the same shape as the tensor (ind)
        # (double every label in the row --> so every row will contain k elements/k columns) 
        correct = ind.eq(targets.view(-1, 1).expand_as(ind))   
        # Sum up the correct predictions --> we will now have one value (the sum)
        correct_total = correct.view(-1).float().sum() 
        # Devide by the batch_size and return the percentage 
        return correct_total.item() * (100.0 / batch_size)
    
    def _train_one_epoch(
        self,
        epoch: int,
        data_loader: DataLoader,
        criterion: nn.Module,
        encoder_optimizer: torch.optim.Optimizer,
        decoder_optimizer: torch.optim.Optimizer,
        is_finetune: bool = False,
        device: str = "cuda",
    ):
        self.encoder.train()
        self.decoder.train()

        losses = MetricTracker()
        top5_accuracy = MetricTracker()

        pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
        for images, captions, caption_lengths in pbar:
            images = images.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)

            image_features = self.encoder(images)
            predictions, targets, decode_lengths = self.decoder(
                *image_features, 
                encoded_captions=captions, 
                caption_lengths=caption_lengths
            )

            loss: torch.Tensor = criterion(predictions, targets)

            # encoder_optimizer only available when finetune
            if encoder_optimizer:
                encoder_optimizer.zero_grad()

            decoder_optimizer.zero_grad()

            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), 
                self.config.training.grad_clip
            )
            torch.nn.utils.clip_grad_norm_(
                self.decoder.parameters(), 
                self.config.training.grad_clip
            )

            # encoder_optimizer only available when finetune
            if encoder_optimizer:
                encoder_optimizer.step()

            decoder_optimizer.step()

            top5 = self._calculate_accuracy(predictions, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5_accuracy.update(top5, sum(decode_lengths))

            pbar.set_postfix(
                loss=losses.avg,
                top5_accuracy=top5_accuracy.avg,
                finetune=is_finetune
            )

        return losses.avg, top5_accuracy.avg

    def _initialize_criterion(self) -> nn.Module:
        return getattr(torch.nn, self.config.training.loss.name)(
            **self.config.training.loss.config
        )
    
    def _initialize_optimizer(
        self, 
        model: nn.Module, 
        config: OptimizerConfig
    ) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        params = filter(lambda p: p.requires_grad, model.parameters())
        
        # check if params empty
        if not list(params):
            return None, None
        
        optimizer = getattr(torch.optim, config.name)(
            filter(lambda p: p.requires_grad, model.parameters()), 
            **config.config
        )

        lr_scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler.name)(
            optimizer,
            **config.lr_scheduler.config
        )
        

        return optimizer, lr_scheduler

    def _load_checkpoint(
        self,
        checkpoint: str,
        encoder_optimizer: torch.optim.Optimizer,
        decoder_optimizer: torch.optim.Optimizer,
        encoder_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        decoder_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> CheckpointData:
        print(f"Loading checkpoint from {checkpoint}")
        ckpt = torch.load(checkpoint)
        is_finetune = ckpt.get("is_finetune", False)

        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        
        if is_finetune and ckpt["encoder_optimizer"] and ckpt["encoder_lr_scheduler"]:
            self.encoder.set_finetune_layer(self.config.training.finetune_n_layer)

            # Reinitialize the optimizer and scheduler
            encoder_optimizer, encoder_lr_scheduler = self._initialize_optimizer(
                self.encoder, self.config.training.encoder_optimizer
            )

            # encoder optimizer only available when finetune
            # since, we freeze all encoder on the first N - k epochs
            encoder_optimizer.load_state_dict(ckpt["encoder_optimizer"])
            encoder_lr_scheduler.load_state_dict(ckpt["encoder_lr_scheduler"])

        decoder_optimizer.load_state_dict(ckpt["decoder_optimizer"])
        decoder_lr_scheduler.load_state_dict(ckpt["decoder_lr_scheduler"])

        return CheckpointData(
            epoch=ckpt["epoch"] + 1,
            best_score=ckpt["best_score"],
            n_epoch_no_improvement=ckpt["n_epoch_no_improvement"],
            is_finetune=is_finetune
        )

    def _initialize_train_dataloader(self) -> DataLoader:
        dataset = COCOTextDataset(
            self.config.data.train,
            self.tokenizer,
            transform=self.get_transform(self.config)
        )

        data_loader = DataLoader(
            dataset, 
            batch_size=self.config.data.batch_size, 
            num_workers=self.config.data.num_workers,
            shuffle=True
        )

        return data_loader

    def train(self, device: str = "cuda", checkpoint: str = None, **kwargs):
        """
        Out training pipeline
        1. Freeze encoder, and train decoder only for N - k epochs
        2. Unfreeze encoder, and train both encoder and decoder for k epochs
        """

        # Step 1. Freeze encoder, and train decoder only for N - k epochs
        set_requires_grad(self.encoder, False)
        set_requires_grad(self.decoder, True)

        criterion = self._initialize_criterion()

        encoder_optimizer, encoder_lr_scheduler = self._initialize_optimizer(
            self.encoder, self.config.training.encoder_optimizer
        )
        decoder_optimizer, decoder_lr_scheduler = self._initialize_optimizer(
            self.decoder, self.config.training.decoder_optimizer
        )

        start_epoch = 0
        best_score = 0
        n_epoch_no_improvement = 0
        is_finetune = False

        if checkpoint:
            checkpoint_data = self._load_checkpoint(
                checkpoint,
                encoder_optimizer,
                decoder_optimizer,
                encoder_lr_scheduler,
                decoder_lr_scheduler
            )
            start_epoch = checkpoint_data.epoch
            best_score = checkpoint_data.best_score
            n_epoch_no_improvement = checkpoint_data.n_epoch_no_improvement
            is_finetune = checkpoint_data.is_finetune

        # encoder_optimizer and scheduler only available when finetune
        if encoder_optimizer and encoder_lr_scheduler:
            optimizer_to(encoder_optimizer, device)
            scheduler_to(encoder_lr_scheduler, device)

        optimizer_to(decoder_optimizer, device)
        scheduler_to(decoder_lr_scheduler, device)

        self.encoder.to(device)
        self.decoder.to(device)

        data_loader = self._initialize_train_dataloader()
        
        for epoch in range(start_epoch, self.config.training.epochs):
            if epoch == self.config.training.epochs - self.config.training.finetune_epochs:
                # Step 2. Unfreeze encoder, and train both encoder and decoder for k epochs
                print(f"Start finetuning the encoder with {self.config.training.finetune_n_layer} layers.")
                is_finetune = True
                self.encoder.set_finetune_layer(self.config.training.finetune_n_layer)

                # Reinitialize the optimizer and scheduler
                encoder_optimizer, encoder_lr_scheduler = self._initialize_optimizer(
                    self.encoder, self.config.training.encoder_optimizer
                )

            loss, top5_accuracy = self._train_one_epoch(
                epoch, 
                data_loader, 
                criterion, 
                encoder_optimizer, 
                decoder_optimizer, 
                device=device,
                is_finetune=is_finetune
            )

            scores = self.evaluate(device=device, split="validation", **kwargs)
            scores = {k: v for k, v in scores.items() if k != "loss"}
            metrics = {
                "loss": loss,
                "top5_accuracy": top5_accuracy,
                **scores
            }
            
            print("Validation scores:", scores)
            score = metrics[self.config.training.track_metric]

            if encoder_lr_scheduler:
                encoder_lr_scheduler.step(score)

            decoder_lr_scheduler.step(score)

            print(f"Encoder LR: {encoder_lr_scheduler.get_last_lr() if encoder_lr_scheduler else 'Not available'}")
            print(f"Decoder LR: {decoder_lr_scheduler.get_last_lr()}")

            if self.config.training.track_metric == "loss":
                score = -score
                
            is_best = score > best_score
            best_score = max(score, best_score)

            self.save_checkpoint(
                epoch=epoch,
                encoder=self.encoder, 
                decoder=self.decoder, 
                encoder_optimizer=encoder_optimizer, 
                decoder_optimizer=decoder_optimizer,
                encoder_lr_scheduler=encoder_lr_scheduler,
                decoder_lr_scheduler=decoder_lr_scheduler,
                metrics=metrics,
                best_score=best_score,
                n_epoch_no_improvement=n_epoch_no_improvement,
                is_best=is_best,
                is_finetune=is_finetune
            )

            if is_best:
                n_epoch_no_improvement = 0
            else:
                n_epoch_no_improvement += 1

            if n_epoch_no_improvement >= self.config.training.early_stop_n_epoch:
                print(f"Early stopping at epoch {epoch}")
                break
            
    def save_checkpoint(
        self, 
        epoch: int,
        encoder: nn.Module, 
        decoder: nn.Module, 
        encoder_optimizer: torch.optim.Optimizer, 
        decoder_optimizer: torch.optim.Optimizer,
        encoder_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        decoder_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        metrics: dict,
        best_score: float,
        n_epoch_no_improvement: int,
        is_finetune: bool,
        is_best: bool = False,
    ):
        state = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "encoder_optimizer": encoder_optimizer.state_dict() if encoder_optimizer else None,
            "decoder_optimizer": decoder_optimizer.state_dict(),
            "metrics": metrics,
            "encoder_lr_scheduler": encoder_lr_scheduler.state_dict() if encoder_lr_scheduler else None,
            "decoder_lr_scheduler": decoder_lr_scheduler.state_dict(),
            "best_score": best_score,
            "n_epoch_no_improvement": n_epoch_no_improvement,
            "is_finetune": is_finetune,
            "config": self.config.model_dump()
        }

        metric_name = "_".join([f"{k}={v:.4f}" for k, v in metrics.items()])

        filename = f"checkpoint_{epoch}_{metric_name}.ckpt"

        if is_best:
            filename = f"BEST_{filename}"

        if not os.path.exists(self.config.training.checkpoint_dir):
            os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)

        output_path = os.path.join(self.config.training.checkpoint_dir, filename)
        
        torch.save(state, output_path)


class MetricTracker:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
