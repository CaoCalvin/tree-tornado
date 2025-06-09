# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: torch-env
#     language: python
#     name: python3
# ---

# %%
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset as BaseDataset
import torch.cuda as cuda
import torch
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from enum import Enum
import cv2 
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler
import math
from pytorch_lightning.callbacks import RichProgressBar
import logging
from itertools import chain, combinations
import wandb
import os
class Cls(Enum):
    UPRIGHT = (0, "#b7f2a6")  # light green
    FALLEN = (1, "#c71933")   # red
    OTHER = (2, "#ffcc33")    # yellow

    def __new__(cls, num, hex_color):
        obj = object.__new__(cls)
        obj._value_ = num
        obj.hex_color = hex_color
        obj.rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))  # Convert hex to RGB
        return obj
    
def resplit(dataset_path, train_frac=0.7, val_frac=0.2, test_frac=0.1, seed=42):
    assert train_frac + val_frac + test_frac == 1.0, "Fractions must sum to 1."

    scenes = sorted(os.listdir(os.path.join(dataset_path, 'images')))
    random.seed(seed)
    random.shuffle(scenes)
    
    n = len(scenes)
    n_train = int(n * train_frac)
    # Use remaining scenes for both validation and test equally
    remaining = n - n_train
    half = remaining // 2  # ensure both splits have the same number of scenes
    
    train_scenes = scenes[:n_train]
    val_scenes = scenes[n_train:n_train+half]
    test_scenes = scenes[n_train+half:n_train+2*half]

    os.makedirs(os.path.join(dataset_path, 'splits'), exist_ok=True)
    for name, split in zip(['train', 'val', 'test'], [train_scenes, val_scenes, test_scenes]):
        with open(os.path.join(dataset_path, 'splits', f'{name}.txt'), 'w') as f:
            for scene in split:
                f.write(f"{scene}\n")

class Dataset(BaseDataset):

    def __init__(self, image_root, mask_root, split_file, transform=None):
        self.background_class = Cls.UPRIGHT.value
        assert os.path.exists(image_root), f"Image root {image_root} does not exist."
        assert os.path.exists(mask_root), f"Mask root {mask_root} does not exist."
        assert os.path.exists(split_file), f"Split file {split_file} does not exist."
        
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        
        with open(split_file, 'r') as f:
            self.scenes = [line.strip() for line in f if line.strip()]
        
        self.samples = []
        for scene in self.scenes:
            image_dir = os.path.join(self.image_root, scene)
            mask_dir = os.path.join(self.mask_root, scene)
            
            for fname in sorted(os.listdir(image_dir)):
                if not fname.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                    continue
                img_path = os.path.join(image_dir, fname)
                name, ext = os.path.splitext(fname)
                mask_path = os.path.join(mask_dir, f"{name}_mask{ext}")  
                self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        return image, mask
   
def get_training_augmentation(input_size=512):
    """
    An augmentation pipeline that avoids padding artifacts by applying
    geometric distortions first, then using RandomResizedCrop to select a
    clean patch and resize it to the target size.

    Args:
        input_size (int): The target height and width of the image.

    Returns:
        A.Compose: The Albumentations transformation pipeline.
    """
    transform = A.Compose([
        # 1. Start with heavy geometric distortions.
        # These transforms can introduce blank areas at the borders (artifacts).
        # We will remove these artifacts in the next step.
        A.ShiftScaleRotate(
            shift_limit=0.0625,  # Max shift of 6.25% of image dimension
            scale_limit=0.1,     # Zoom in or out by up to 10%
            rotate_limit=10,     # Rotate by up to 10 degrees
            p=0.7,
            border_mode=cv2.BORDER_REFLECT_101, # The padding introduced here...
        ),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=35, sigma=120 * 0.05, alpha_affine=120 * 0.03,
                               border_mode=cv2.BORDER_REFLECT_101),
            A.GridDistortion(p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        ], p=0.3),

        # Crop a random part and resize it back.
        A.RandomResizedCrop(
            height=input_size,
            width=input_size,
            scale=(0.9, 1.0),
            p=.5 
        ),

        # 3. Apply simple, non-distorting augmentations.
        # These are safe to apply now because the image size is fixed at 512x512.
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # 4. Color, blur, and noise augmentations.
        # These should generally come after all geometric changes.
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=15, p=0.3),
        A.OneOf([
            A.GaussianBlur(p=0.5),
            A.MotionBlur(p=0.5),
        ], p=0.3),
        A.GaussNoise(p=0.2),

        # 5. Final normalization and conversion to a tensor.
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    try:
        # Log the string representation of the new, improved pipeline.
        wandb_logger.log({"training_transforms": str(transform)})
    except Exception as e:
        print("W&B logging not available (or not initialized):", e)
    return transform, str(transform)


def get_validation_augmentation():
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return transform, str(transform)

def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, img) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if name == "image":
            # Check if image is in CHW format; if so, convert it to HWC for plotting
            if isinstance(img, torch.Tensor):
                if img.shape[0] == 3:
                    img = img.permute(1, 2, 0).cpu().numpy()
            else:
                if img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
            # Remove ImageNet normalization by inverting it
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            plt.imshow(img)
        else:
            # Convert greyscale integer mask to RGB colored mask using the enum directly.
            color_mask = np.zeros((*img.shape, 3), dtype=np.uint8)
            for cls_member in Cls:
                color_mask[img == cls_member.value] = cls_member.rgb_color
            plt.imshow(color_mask)

class ImageLoggingCallback(pl.Callback):
    """
    Logs a batch of validation samples to W&B, including the input image,
    ground truth mask, and the model's predicted mask.

    This callback finds a specified number of images for each unique
    combination of classes present in the ground truth masks of the
    validation set.
    """
    # --- MODIFICATION START ---
    def __init__(self, num_samples_per_combo=1):
        """
        Args:
            num_samples_per_combo (int): The number of images to log for each
                                         unique class combination.
        """
        super().__init__()
        # Store the number of samples to log for each combination
        self.num_samples_per_combo = num_samples_per_combo
        self.class_labels = {cls.value: cls.name for cls in Cls}
        
        # Dynamically generate all possible non-empty subsets of class values
        class_values = [cls.value for cls in Cls]
        all_combinations = chain.from_iterable(combinations(class_values, r) for r in range(1, len(class_values) + 1))
        self.target_combinations = {frozenset(combo) for combo in all_combinations}
        
        # Use a dictionary to count how many we've logged for each combination
        self.combination_counts = {}
    # --- MODIFICATION END ---

    def _is_done(self):
        """Checks if we have logged the desired number of samples for all combinations."""
        if len(self.combination_counts) < len(self.target_combinations):
            return False
        
        return all(count >= self.num_samples_per_combo for count in self.combination_counts.values())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only log images at the end of the first validation epoch
        if trainer.current_epoch != 0:
            return

        val_loader = trainer.val_dataloaders[0]
        device = pl_module.device
        images_to_log = []

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        pl_module.eval()
        with torch.no_grad():
            for batch in val_loader:
                # Break early if we've found enough images for every combination
                if self._is_done():
                    break

                images, gt_masks = batch
                images, gt_masks = images.to(device), gt_masks.to(device)
                logits = pl_module(images)
                pred_masks = torch.argmax(logits, dim=1)

                for i in range(images.shape[0]):
                    gt_mask = gt_masks[i]
                    present_classes = frozenset(torch.unique(gt_mask).cpu().numpy())

                    # --- MODIFICATION START ---
                    # If this is a target combination and we haven't logged enough samples yet
                    current_count = self.combination_counts.get(present_classes, 0)
                    if present_classes in self.target_combinations and current_count < self.num_samples_per_combo:
                        # Increment count for this combination
                        self.combination_counts[present_classes] = current_count + 1
                    # --- MODIFICATION END ---

                        image_vis = (images[i].unsqueeze(0) * std + mean).clamp(0, 1)
                        image_vis_np = (image_vis.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                        gt_mask_np = gt_mask.cpu().numpy().astype(np.uint8)
                        pred_mask_np = pred_masks[i].cpu().numpy().astype(np.uint8)
                        
                        # --- MODIFICATION START ---
                        # Create a more detailed caption including the sample number
                        class_names = [self.class_labels[c] for c in sorted(list(present_classes))]
                        caption = f"Classes: {class_names}, Sample #{current_count + 1}"
                        # --- MODIFICATION END ---

                        wandb_image = wandb.Image(
                            image_vis_np,
                            caption=caption,
                            masks={
                                "ground_truth": {"mask_data": gt_mask_np, "class_labels": self.class_labels},
                                "prediction": {"mask_data": pred_mask_np, "class_labels": self.class_labels},
                            },
                        )
                        images_to_log.append(wandb_image)

        if images_to_log:
            # Sort by caption to have a consistent order in the UI
            images_to_log.sort(key=lambda img: img.caption)
            trainer.logger.experiment.log({"Validation Samples": images_to_log})

class CamVidModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        # A list of class names derived from the Cls enum for logging purposes.
        self.class_names = [c.name for c in Cls]
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # In a real application, you might want to un-normalize the image
        # before passing it to the model if your augmentations include normalization.
        # image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage, batch_idx):
        image, mask = batch
        image = image.to(device)
        mask = mask.to(device)
        assert image.ndim == 4, "Expected image to have 4 dimensions, got shape " + str(image.shape)
        mask = mask.long()
        assert mask.ndim == 3, "Expected mask to have 3 dimensions, got shape " + str(mask.shape)
        logits_mask = self.forward(image)
        assert logits_mask.shape[1] == self.number_of_classes, f"Expected logits channel {self.number_of_classes}, got {logits_mask.shape[1]}"
        logits_mask = logits_mask.contiguous()
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.softmax(dim=1)
        pred_mask = prob_mask.argmax(dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    def shared_epoch_end(self, outputs, stage):
        # Aggregate stats from all batches
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        # print(f"[DEBUG] {stage} epoch end: Aggregated tp shape: {tp.shape}, fp shape: {fp.shape}, fn shape: {fn.shape}, tn shape: {tn.shape}")

        # Calculate overall metrics
        metrics = {
            f"{stage}_per_image_iou": smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise"),
            f"{stage}_dataset_iou_micro": smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro"),
            f"{stage}_dataset_iou_macro": smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro"),
        }
        # print(f"[DEBUG] {stage} epoch end: Overall metrics calculated: {metrics}")

        # --- START OF FIX ---
        # To get the IoU per class for the whole dataset, we must sum the stats along the sample dimension (dim=0) first.
        tp_agg = torch.sum(tp, dim=0)
        fp_agg = torch.sum(fp, dim=0)
        fn_agg = torch.sum(fn, dim=0)
        tn_agg = torch.sum(tn, dim=0)

        # This will now return a tensor of shape (C,), where C is the number of classes.
        per_class_ious = smp.metrics.iou_score(tp_agg, fp_agg, fn_agg, tn_agg, reduction=None)
        # --- END OF FIX ---

        # print(f"[DEBUG] {stage} epoch end: Per-class IoUs raw output: {per_class_ious}")
        for i, iou in enumerate(per_class_ious):
            class_name = self.class_names[i]
            # 'iou' is now a scalar tensor, so the .mean() part is not strictly necessary but doesn't hurt.
            class_iou = iou.mean() if iou.numel() > 1 else iou
            metrics[f"{stage}_iou_class_{class_name}"] = class_iou
            # print(f"[DEBUG] {stage} epoch end: IoU for class '{class_name}' is: {class_iou}")

        self.log_dict(metrics, prog_bar=True)
        # print(f"[DEBUG] {stage} epoch end: Final logged metrics: {metrics}")

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train", batch_idx)
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid", batch_idx)
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test", batch_idx)
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = OPTIMIZER_TYPE(self.parameters(), lr=LEARNING_RATE)
        scheduler = SCHEDULER_TYPE(optimizer, T_max=T_MAX, eta_min=ETA_MIN)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
        },
}
# %% [markdown]
# # Create model and train

# %%

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn', force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")

    
    base_path = os.path.join('..', 'dataset_processed')

    # Assert all images 512x512
    # Check that all images under the dataset_processed images directory are 512x512
    images_dir = os.path.join(base_path, 'images')
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    if img.size != (512, 512):
                        raise AssertionError(f"Image {img_path} has size {img.size}, expected (512, 512)")

    images_path = os.path.join(base_path, 'images')
    masks_path = os.path.join(base_path, 'masks')
    splits_path = os.path.join(base_path, 'splits')

    
    # resplit(base_path, train_frac=0.7, val_frac=0.15, test_frac=0.15)

    training_transform_obj, training_transform_str = get_training_augmentation()
    dataset_train = Dataset(
        image_root=images_path,
        mask_root=masks_path,
        split_file=os.path.join(splits_path, 'train.txt'),
        transform=training_transform_obj
    )

    validation_transform_obj, validation_transform_str = get_validation_augmentation()

    dataset_val = Dataset(
        image_root=images_path,
        mask_root=masks_path,
        split_file=os.path.join(splits_path, 'val.txt'),
        transform=validation_transform_obj  
    )

    dataset_test = Dataset(
        image_root=images_path,
        mask_root=masks_path,
        split_file=os.path.join(splits_path, 'test.txt'),
        transform=validation_transform_obj  
    )


    # Some training hyperparameters TODO tune
    EPOCHS = 35
    BATCH_SIZE = 16
    T_MAX = EPOCHS * math.ceil(len(dataset_train) / BATCH_SIZE)
    OUT_CLASSES = len(Cls)

    # Optimizer and scheduler parameters
    OPTIMIZER_TYPE = torch.optim.Adam
    LEARNING_RATE = 2e-4
    SCHEDULER_TYPE = lr_scheduler.CosineAnnealingLR
    ETA_MIN = 1e-5

    # Architecture and encoder parameters
    ARCH = "SegFormer"
    ENCODER_NAME = "mit_b3"
    ENCODER_WEIGHTS = "imagenet"

    SEED = 42

    idx = random.randint(0, len(dataset_train) - 1)
    image, mask = dataset_train[idx] 
    print(f"Showing image {idx} of {len(dataset_train)}")
    print(f"Mask shape: {mask.shape}")
    print(f"Image shape: {image.shape}")
    visualize(image=image, mask=mask)


    torch.set_float32_matmul_precision('medium') # TODO see if high is better or low doesn't make a difference

    wandb_logger = WandbLogger(project="treefall-tornado-rating", log_model=True)

    model = CamVidModel(
        ARCH,
        ENCODER_NAME,
        in_channels=3,
        out_classes=OUT_CLASSES,
        encoder_weights=ENCODER_WEIGHTS
    ).to(device)
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    wandb_logger.experiment.config.update({
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "t_max": T_MAX,
        "optimizer": OPTIMIZER_TYPE.__name__,
        "lr": LEARNING_RATE,
        "scheduler": SCHEDULER_TYPE.__name__,
        "min_lr": ETA_MIN,
        "architecture": ARCH,
        "encoder": ENCODER_NAME,
        "encoder_weights": ENCODER_WEIGHTS,
        "train_size": len(dataset_train),
        "val_size": len(dataset_val),
        "num_classes": OUT_CLASSES,
        "training_transform": training_transform_str,
        "validation_transform": validation_transform_str,
    })

    # TODO: Tune number of workers based on system
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, persistent_workers=True, pin_memory=True)

    image_logging_callback = ImageLoggingCallback(num_samples_per_combo=3)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        log_every_n_steps=1,
        fast_dev_run=False,
        callbacks=[RichProgressBar(), image_logging_callback], 
        logger=wandb_logger
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
