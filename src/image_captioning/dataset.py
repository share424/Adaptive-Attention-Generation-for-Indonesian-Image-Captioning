"""Dataset for image captioning"""
from typing import Tuple, List
import os

from pycocotools.coco import COCO
from torch.utils.data import Dataset
from PIL import Image
from torch import Tensor, LongTensor

from .tokenizer import Tokenizer
from .config import CocoConfig


class COCOTextDataset(Dataset):
    def __init__(
        self, 
        configs: List[CocoConfig],
        tokenizer: Tokenizer, 
        transform=None,
        return_raw_caption: bool = False
    ):
        self.cocos = [COCO(c.annotation) for c in configs]
        self.transform = transform
        self.image_dirs = [c.image_dir for c in configs]
        self.tokenizer = tokenizer
        self.return_raw_caption = return_raw_caption
        self.annotation_index = []
        self._index_annotation()

    def _index_annotation(self):
        """Index the annotations for fast retrieval"""
        for i, coco in enumerate(self.cocos):
            for ann_id in coco.getAnnIds():
                self.annotation_index.append((i, ann_id))

    def __len__(self):
        return len(self.annotation_index)
    
    def _get_annotation(self, idx: int):
        coco_idx, ann_id = self.annotation_index[idx]
        return coco_idx, self.cocos[coco_idx].loadAnns(ann_id)[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get item by index
        
        Args:
            idx (int): index

        Returns:
            tuple[Tensor, Tensor, Tensor]: image, caption, caption_length
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range. Max: {len(self)}")
        
        coco_idx, ann = self._get_annotation(idx)
        caption_str: str = ann['caption']
        img_id = ann['image_id']
        filename = self.cocos[coco_idx].loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.image_dirs[coco_idx], filename)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        caption, caption_length = self.tokenizer.tokenize(caption_str, return_length=True)
        
        output = [image, LongTensor(caption), LongTensor(caption_length)]
        if self.return_raw_caption:
            output.append(caption_str)

        return tuple(output)


class COCOImageDataset(Dataset):
    def __init__(
        self, 
        configs: List[CocoConfig],
        tokenizer: Tokenizer, 
        return_keys: List[str],
        transform=None,
        process_captions: bool = False,
    ):
        self.cocos = [COCO(c.annotation) for c in configs]
        self.transform = transform
        self.image_dirs = [c.image_dir for c in configs]
        self.tokenizer = tokenizer
        self.return_keys = return_keys
        self.return_captions = process_captions
        self.annotation_index = []
        self._index_annotation()

    def _index_annotation(self):
        """Index the annotations for fast retrieval"""
        for i, coco in enumerate(self.cocos):
            for image_id in coco.getImgIds():
                self.annotation_index.append((i, image_id))

    def __len__(self):
        return len(self.annotation_index)
    
    def _get_annotation(self, idx: int):
        coco_idx, image_id = self.annotation_index[idx]
        return coco_idx, self.cocos[coco_idx].loadImgs(image_id)[0]
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get item by index
        
        Args:
            idx (int): index

        Returns:
            tuple[Tensor, Tensor, Tensor]: image, caption, caption_length
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range. Max: {len(self)}")
        
        coco_idx, img = self._get_annotation(idx)
        filename = img['file_name']
        image_id = img['id']
        image = Image.open(os.path.join(self.image_dirs[coco_idx], filename)).convert("RGB")
        if self.transform:
            image = self.transform(image)

        data = {
            "image": image,
            "image_id": image_id,
        }
        
        if self.return_captions:
            captions = self.cocos[coco_idx].loadAnns(self.cocos[coco_idx].getAnnIds(imgIds=img['id']))
            captions_str = [c['caption'] for c in captions]
            captions, caption_lengths = self.tokenizer.tokenize(captions_str, return_length=True)

            data["captions"] = LongTensor(captions)
            data["caption_lengths"] = LongTensor(caption_lengths)
            data["captions_str"] = captions_str

        output = [data[k] for k in self.return_keys]

        return tuple(output)