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
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler
import math
from lightning.pytorch.callbacks import RichProgressBar
import logging
from itertools import chain, combinations
import wandb
import os
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torchmetrics.classification import MulticlassJaccardIndex

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
    
import logging

def resplit(dataset_path, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    """Splits scene folders into train, validation, and test sets."""
    assert math.isclose(train_frac + val_frac + test_frac, 1.0), "Fractions must sum to 1."
    
    images_path = os.path.join(dataset_path, 'images')
    assert os.path.exists(images_path), f"Dataset images path does not exist: {images_path}"
    
    scenes = sorted(os.listdir(images_path))
    assert scenes, f"No scenes found in {images_path}. The directory is empty."
    
    random.shuffle(scenes)

    n = len(scenes)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_scenes = scenes[:n_train]
    val_scenes = scenes[n_train : n_train + n_val]
    test_scenes = scenes[n_train + n_val:] # Assign the rest to test set


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

        assert self.scenes, f"Split file is empty: {split_file}"

        self.samples = []
        for scene in self.scenes:
            image_dir = os.path.join(self.image_root, scene)
            mask_dir = os.path.join(self.mask_root, scene)
            
            assert os.path.isdir(image_dir), f"Scene directory not found in images: {image_dir}"
            assert os.path.isdir(mask_dir), f"Scene directory not found in masks: {mask_dir}"

            for fname in sorted(os.listdir(image_dir)):
                if not fname.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                    continue
                img_path = os.path.join(image_dir, fname)
                name, ext = os.path.splitext(fname)
                mask_path = os.path.join(mask_dir, f"{name}_mask{ext}")

                # CRITICAL: Ensure every image has a corresponding mask before adding to samples.
                assert os.path.exists(mask_path), f"Mask not found for image: {img_path}\nExpected at: {mask_path}"
                
                self.samples.append((img_path, mask_path))

        assert len(self.samples) > 0, f"No samples were found for the split defined by {split_file}. Check file paths and content."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))

        # Check dimensions before augmentation
        assert image.ndim == 3, f"Image should have 3 dimensions (H, W, C), but got {image.ndim} for {img_path}"
        assert mask.ndim == 2, f"Mask should have 2 dimensions (H, W), but got {mask.ndim} for {mask_path}"
        assert image.shape[:2] == mask.shape, f"Image and mask dimensions do not match: {image.shape[:2]} vs {mask.shape} for {img_path}"

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        
        # Check data types and shapes after augmentation
        assert isinstance(image, torch.Tensor), f"Image should be a torch.Tensor but got {type(image)}"
        assert isinstance(mask, torch.Tensor), f"Mask should be a torch.Tensor but got {type(mask)}"
        assert image.dtype == torch.float32, f"Image tensor should have dtype float32, but has {image.dtype}"
        assert mask.dtype == torch.int64 or mask.dtype == torch.long or mask.dtype == torch.uint8, f"Mask tensor should have dtype int64/long, but has {mask.dtype}"
        
        return image, mask, img_path
   
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
            size=(input_size, input_size),
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
    return transform, str(transform)


def get_validation_augmentation():
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return transform, str(transform)

def visualize(image, gt_mask, pred_mask):
    """
    Creates a side-by-side visualization of ground truth and prediction overlays.

    Args:
        image (torch.Tensor): The input image tensor (C, H, W).
        gt_mask (torch.Tensor): The ground truth mask tensor (H, W).
        pred_mask (torch.Tensor): The predicted mask tensor (H, W).

    Returns:
        np.array: A single RGB image array (uint8) showing the side-by-side overlays.
    """
    # 1. Denormalize image and convert to a displayable NumPy array
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    image_vis = (image * std + mean).clamp(0, 1)
    image_np = (image_vis.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    # Convert masks to NumPy arrays
    gt_mask_np = gt_mask.cpu().numpy().astype(np.uint8)
    pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)

    # 2. Helper function to create a single overlay
    def _create_overlay(img, mask):
        color_mask = np.zeros_like(img, dtype=np.uint8)
        for cls_member in Cls:
            color_mask[mask == cls_member.value] = cls_member.rgb_color
        return cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)

    # 3. Create overlays for ground truth and prediction
    gt_overlay = _create_overlay(image_np, gt_mask_np)
    pred_overlay = _create_overlay(image_np, pred_mask_np)

    # 4. Concatenate side-by-side and return
    return np.concatenate([gt_overlay, pred_overlay], axis=1)


class ImageLoggingCallback(pl.Callback):
    """
    Logs a batch of validation samples to W&B, including side-by-side overlays
    of the image with the ground truth mask (left) and the image with the predicted mask (right).
    """
    def __init__(self, num_samples_per_combo=1, img_logging_interval=5):
        """
        Args:
            num_samples_per_combo (int): The number of images to log for each
                                         unique class combination.
        """
        super().__init__()
        self.num_samples_per_combo = num_samples_per_combo
        self.class_labels = {cls.value: cls.name for cls in Cls}
        self.class_colors = {cls.value: cls.rgb_color for cls in Cls}  # to get color info
        
        # Dynamically generate all possible non-empty subsets of class values
        class_values = [cls.value for cls in Cls]
        all_combinations = chain.from_iterable(combinations(class_values, r) for r in range(1, len(class_values) + 1))
        self.target_combinations = {frozenset(combo) for combo in all_combinations}
        
        # Counter for each combination
        self.combination_counts = {}

    def _create_overlay(self, image, mask):
        """
        Creates an overlay of the mask on top of the image.
        Args:
            image (np.array): The image array (H, W, 3) as uint8.
            mask (np.array): The mask array (H, W) as uint8.
        Returns:
            np.array: The blended overlay image.
        """
        # Create a blank color mask
        color_mask = np.zeros_like(image)
        for class_val, color in self.class_colors.items():
            color_mask[mask == class_val] = np.array(color, dtype=np.uint8)
        # Blend the original image with the color mask.
        overlay = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)
        return overlay

    def _is_done(self):
        """Checks if we have logged the desired number of samples for all combinations."""
        if len(self.combination_counts) < len(self.target_combinations):
            return False
        return all(count >= self.num_samples_per_combo for count in self.combination_counts.values())
    
    def on_validation_epoch_start(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self.combination_counts = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip during sanity check
        if trainer.sanity_checking:
            return

        if trainer.current_epoch % img_logging_interval != 0:
            return

        # Correctly select the validation dataloader
        val_dataloaders = trainer.val_dataloaders
        val_loader = val_dataloaders[0] if isinstance(val_dataloaders, list) else val_dataloaders

        device = pl_module.device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        pl_module.eval()
        with torch.no_grad():
            for batch in val_loader:
                if self._is_done():
                    break

                # Unpack the batch and send to device
                images, gt_masks, img_paths = batch
                images, gt_masks = images.to(device), gt_masks.to(device)
                logits = pl_module(images)
                pred_masks = torch.argmax(logits, dim=1)


                # Create randomly ordered indices to shuffle the images
                indices = torch.randperm(images.shape[0])

                for i in range(indices):
                    gt_mask = gt_masks[i]
                    present_classes = frozenset(torch.unique(gt_mask).cpu().numpy())
                    current_count = self.combination_counts.get(present_classes, 0)

                    if present_classes in self.target_combinations and current_count < self.num_samples_per_combo:
                        self.combination_counts[present_classes] = current_count + 1

                        # Use the new, clean visualize function
                        combined_overlay = visualize(
                            image=images[i],
                            gt_mask=gt_mask,
                            pred_mask=pred_masks[i]
                        )

                        filename = os.path.basename(img_paths[i])
                        class_names = [self.class_labels[c] for c in sorted(list(present_classes))]
                        caption = f"GT (left) | Pred (right) | Classes: {class_names} | Sample #{current_count + 1}"

                        wandb_image = wandb.Image(
                            combined_overlay,
                            caption=caption,
                            file_type="jpg",
                        )
                        
                        trainer.logger.experiment.log({f"Validation Samples/Epoch {trainer.current_epoch}/{filename}": wandb_image})

                        # If we have logged all required images, we can stop iterating
                        if self._is_done():
                            return

class CamVidModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, lr, optimizer_type, scheduler_type, weight_decay, drop_path_rate, warmup_steps, t_max, eta_min, freeze_encoder, **kwargs):
        super().__init__()
        self.save_hyperparameters() # This is a handy PyTorch Lightning feature to save all constructor arguments

        # The encoder_config is used to set the drop_path_rate for SegFormer
        encoder_config = {
            "drop_path_rate": self.hparams.drop_path_rate,
        }

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            encoder_weights="imagenet",
            encoder_config=encoder_config,
            **kwargs,
        )

        # Jaccard Index is the official name for IoU (Intersection over Union)
        self.iou = MulticlassJaccardIndex(num_classes=out_classes, average='none')

        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes
        self.class_names = [c.name for c in Cls]
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        return self.model(image)

    def _common_step(self, batch, batch_idx, stage):
        image, mask, _ = batch
        mask = mask.long()

        # Get the batch size directly from the input tensor
        batch_size = image.shape[0]

        logits = self.forward(image)
        loss = self.loss_fn(logits, mask)
        
        # Pass the batch_size to self.log()
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        pred_mask = torch.argmax(logits, dim=1)
        self.iou.update(pred_mask, mask)
 
        per_class_iou = self.iou.compute()  # returns tensor with per-class IoU

        # Log overall IoU as the mean of per-class metrics
        overall_iou = per_class_iou.mean()
        self.log(f"{stage}_iou_overall", overall_iou,
                on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # Log per-class IoU
        self.log_dict({f"{stage}_iou_class_{self.class_names[i]}": per_class_iou[i]
                    for i in range(len(per_class_iou))},
                    on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "valid")

    def configure_optimizers(self):
        # We pass the weight_decay to the optimizer

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)



        if self.hparams.scheduler_type == 'CosineAnnealingLR':

            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.t_max, eta_min=self.hparams.eta_min)

        elif self.hparams.scheduler_type == 'OneCycleLR':

            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr, total_steps=self.hparams.t_max)

        else:

            raise ValueError("Unsupported scheduler type")



        # Optional: Add a learning rate warmup scheduler

        if self.hparams.warmup_steps > 0:

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(

                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=self.hparams.warmup_steps

            )

            # Chain the warmup scheduler with the main scheduler

            lr_scheduler_config = {

                "scheduler": torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[self.hparams.warmup_steps]),

                "interval": "step",

                "frequency": 1,

            }

        else:

            lr_scheduler_config = {

                "scheduler": scheduler,

                "interval": "step",

                "frequency": 1,

            }



        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
 
def objective(trial: optuna.Trial):
    # -- 1. Define the hyperparameter search space ---
    # We use trial.suggest_ to define the range and type of each hyperparameter.
    
    """
    # Categorical parameters
    encoder_name = trial.suggest_categorical("encoder_name", ["mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4"])
    scheduler_type = trial.suggest_categorical("scheduler_type", ["CosineAnnealingLR", "OneCycleLR"])

    # Integer parameters
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16]) # Use categorical for specific values
    warmup_steps = trial.suggest_int("warmup_steps", 0, 500)

    # Float parameters
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 0.3)
    """
    # Categorical parameters
    encoder_name = trial.suggest_categorical("encoder_name", ["mit_b0"])
    scheduler_type = trial.suggest_categorical("scheduler_type", ["CosineAnnealingLR"])

    # Integer parameters
    batch_size = trial.suggest_categorical("batch_size", [16]) # Use categorical for specific values
    warmup_steps = trial.suggest_int("warmup_steps", 0, 0)

    # Float parameters
    learning_rate = trial.suggest_float("lr", 1e-3, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-3, log=True)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.1, 0.1)
    


    # --- 2. Setup Datasets and Dataloaders ---
    base_path = os.path.join('..', 'dataset_processed')
    images_path = os.path.join(base_path, 'images')
    masks_path = os.path.join(base_path, 'masks')
    splits_path = os.path.join(base_path, 'splits')
    
    training_transform_obj, _ = get_training_augmentation()
    dataset_train = Dataset(
        image_root=images_path,
        mask_root=masks_path,
        split_file=os.path.join(splits_path, 'train.txt'),
        transform=training_transform_obj
    )

    validation_transform_obj, _ = get_validation_augmentation()
    dataset_val = Dataset(
        image_root=images_path,
        mask_root=masks_path,
        split_file=os.path.join(splits_path, 'val.txt'),
        transform=validation_transform_obj
    )

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2, persistent_workers=True, pin_memory=True)

    # --- 3. Setup Model and Trainer ---
    T_MAX = EPOCHS_MAX * math.ceil(len(dataset_train) / batch_size)
    
    model = CamVidModel(
        arch="SegFormer",
        encoder_name=encoder_name,
        in_channels=3,
        out_classes=len(Cls),
        lr=learning_rate,
        optimizer_type=torch.optim.AdamW, # Using AdamW
        scheduler_type=scheduler_type,
        weight_decay=weight_decay,
        drop_path_rate=drop_path_rate,
        warmup_steps=warmup_steps,
        t_max=T_MAX,
        eta_min=1e-6, # A small minimum learning rate
        freeze_encoder=True # Usually better to finetune the encoder
    )

    # --- 4. Configure Callbacks and Logger ---
    # The Pruning Callback will monitor the validation dataset IoU and stop unpromising trials.
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="valid_iou_overall")
    image_logging_callback = ImageLoggingCallback(num_samples_per_combo=1, img_logging_interval=1)  # Instantiate the callback
    wandb_logger = WandbLogger(project="tree-tornado", group="BOHB-SegFormer", job_type='train')
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS_MAX,
        logger=wandb_logger,
        callbacks=[RichProgressBar(), pruning_callback, image_logging_callback],  # Added callback here
        accelerator="gpu",
        devices=1,
        log_every_n_steps=25,
    )
    
    # --- 5. Run Training and Return the Metric to Optimize ---
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Return the value of the metric that should be maximized.
    # The callback we defined above will report the 'valid_iou_overall' to the trial.
    # We can access it via trial.user_attrs.
    return trainer.callback_metrics["valid_iou_overall"].item()


if __name__ == '__main__':
    # print(pl.__version__)

    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(42)

    EPOCHS_MAX = 55
    
    # --- 1. Create the Optuna Study ---
    # We use the BOHBSampler for efficient searching.
    # The HyperbandPruner is used to stop unpromising trials early.
    study = optuna.create_study(
        study_name="segformer-bohb-tuning",
        direction="maximize", # We want to maximize the validation IoU
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=EPOCHS_MAX, reduction_factor=3
        ),
    )

    # --- 2. Start the optimization ---
    # n_trials is the total number of hyperparameter combinations to test.
    # study.optimize(objective, n_trials=100, timeout=3600*6) # Run for 100 trials or 6 hours
    study.optimize(objective, n_trials=1) # Run for 1 trial
    # --- 3. Print the results ---
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")