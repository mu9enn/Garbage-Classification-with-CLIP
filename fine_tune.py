import os
import wandb
import torchmetrics
import clip
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageFile
from torchvision import transforms as tsfm
import json
import warnings
import argparse
from torchmetrics import AUROC
import copy
from pytorch_lightning import seed_everything
import math
from datetime import datetime
from typing import List, Tuple

# Suppress warnings
warnings.filterwarnings("ignore")

def gettime_str() -> str:
    """
    Gets the current time formatted as a string of pure numbers.

    Returns:
        str: Current time as a string in the format "YYYYMMDDHH".
    """
    now = datetime.now()
    date_time_string = now.strftime("%Y%m%d%H")
    return date_time_string

class AvgMeter:
    """
    Class to compute and store the average, sum, and count of values.
    """

    def __init__(self, name: str = "Metric"):
        self.name = name
        self.reset()

    def reset(self):
        """
        Resets the average, sum, and count to zero.
        """
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val: float, count: int = 1):
        """
        Updates the average, sum, and count with new values.

        Args:
            val (float): The value to add.
            count (int): The count of values to add.
        """
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        """
        Returns a string representation of the current average.

        Returns:
            str: A string showing the name and current average.
        """
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer) -> float:
    """
    Gets the current learning rate from the optimizer.

    Args:
        optimizer: The optimizer to get the learning rate from.

    Returns:
        float: The current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def get_weight_decay_params(model):
    """
    检查模型的参数并将其分类为需要权重衰减的参数和不需要权重衰减的参数。

    参数:
    model: torch.nn.Module
        要检查的模型

    返回:
    p_wd: list
        需要权重衰减的参数列表
    p_non_wd: list
        不需要权重衰减的参数列表
    """
    print("=> checking whether all parameters are set to zero or not")
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    print("p_wd:", len(p_wd))
    print("p_non_wd:", len(p_non_wd))

def pil_loader(path: str) -> Image.Image:
    """
    Loads an image from a file path using PIL.

    Args:
        path (str): The path to the image file.

    Returns:
        Image.Image: The loaded image.
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def create_train_transform(val_transform: tsfm.Compose) -> tsfm.Compose:
    """
    Creates a training transform by adding random data augmentation to a validation transform.

    Args:
        val_transform (tsfm.Compose): The validation transform to augment.

    Returns:
        tsfm.Compose: The new training transform with added augmentations.
    """
    train_transform_list = copy.deepcopy(val_transform.transforms)
    train_transform_list.insert(2, tsfm.RandomVerticalFlip(p=0.3))
    train_transform_list.insert(3, tsfm.RandomHorizontalFlip(p=0.3))
    train_transform_list.insert(4, tsfm.RandomRotation(degrees=15))
    train_transform_list.insert(5, tsfm.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    train_transform = tsfm.Compose(train_transform_list)
    return train_transform

class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, csv_path, mode, diagnosis_map):
        print("==>path:", csv_path)
        print("==>mode:", mode)
        annotations = pd.read_csv(csv_path)
        self.samples = []
        for i in range(len(annotations)):
            image_path = annotations.loc[i, 'path']
            # 提取所有标签列
            labels = annotations.loc[i, ['Harmful_waste', 'Recyclable_waste', 'Food_waste', 'Other_waste']].values
            # 转换为类别索引
            label = labels.argmax()  # 假设每个样本只有一个标签为1
            self.samples.append((image_path, label))
        self.mode = mode
        self.transform = transform

    def __getitem__(self, i):
        image_id, target = self.samples[i]
        path = os.path.join(img_filepath, image_id)
        try:
            img = pil_loader(path)
            image = self.transform(img)
            return image, target
        except IOError as e:
            print(f"Failed to load image {path}: {e}")
            # 返回一个默认的图像和标签，或者跳过
            # 这里返回一个随机图像和标签
            img = Image.new('RGB', (224, 224), color = 'black')
            image = self.transform(img)
            return image, target

    def __len__(self):
        return len(self.samples)

class ProjectionHead(nn.Module):
    def __init__(
            self,
            backbone,
            embedding_dim,
            projection_dim,
            classes,
            dropout=0.2
    ):
        super().__init__()
        self.backbone = backbone
        self.relu = nn.ReLU()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.fc = nn.Linear(projection_dim, classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection(x)
        x = self.fc(x)
        return x

class MLPHead(nn.Module):
    def __init__(
            self,
            backbone,
            embedding_dim,
            projection_dim,
            classes,
            dropout=0.2
    ):
        super().__init__()
        self.backbone = backbone
        self.batch_norm = nn.BatchNorm1d(projection_dim)  # Adding batch normalization
        self.relu = nn.ReLU()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.fc = nn.Linear(projection_dim, classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection(x)
        x = self.batch_norm(x)  # Apply batch normalization before ReLU
        x = self.relu(x)
        x = self.fc(x)
        return x

class LoRALayer_vit(nn.Module):
    def __init__(self, original_layer, rank=16, alpha=1):
        super(LoRALayer_vit, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        if hasattr(original_layer, 'in_proj_weight'):
            in_dim = original_layer.in_proj_weight.shape[1]
            out_dim = original_layer.in_proj_weight.shape[0]
        else:
            self.weight = self.original_layer.weight
            self.bias = self.original_layer.bias
            in_dim = original_layer.weight.shape[1]
            out_dim = original_layer.weight.shape[0]

        self.rank_down = nn.Parameter(torch.zeros(rank, in_dim))
        self.rank_up = nn.Parameter(torch.zeros(out_dim, rank))

        # Initialize the rank_down and rank_up matrices
        nn.init.kaiming_uniform_(self.rank_down, a=math.sqrt(5))
        nn.init.zeros_(self.rank_up)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True):
        # Compute the low-rank update
        low_rank_matrix = self.rank_up @ self.rank_down

        if hasattr(self.original_layer, 'in_proj_weight'):
            adjusted_proj_weight = self.original_layer.in_proj_weight + self.alpha * low_rank_matrix
            self.original_layer.in_proj_weight.data = adjusted_proj_weight
        else:
            adjusted_proj_weight = self.original_layer.weight + self.alpha * low_rank_matrix
            self.original_layer.weight.data = adjusted_proj_weight
        self.weight = adjusted_proj_weight
        output, adjusted_weights = self.original_layer(query, key, value, key_padding_mask=key_padding_mask,
                                                       need_weights=need_weights, attn_mask=attn_mask,
                                                       average_attn_weights=average_attn_weights)

        return (output, adjusted_weights) if need_weights else output

class LoRAConv2d(nn.Module):
    def __init__(self, conv_layer, r=4):
        super(LoRAConv2d, self).__init__()
        self.conv_layer = conv_layer
        self.r = r

        # 获取原始卷积层参数
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size

        # 低秩矩阵参数
        self.A = nn.Parameter(torch.randn(out_channels, r, 1, 1) * 0.01)
        self.B = nn.Parameter(torch.randn(r, in_channels, kernel_size[0], kernel_size[1]) * 0.01)

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        # 计算低秩近似卷积核并加到原始卷积核上
        low_rank_weight = torch.matmul(self.A.view(self.A.size(0), -1), self.B.view(self.B.size(0), -1))
        low_rank_weight = low_rank_weight.view(self.A.size(0), self.B.size(1), self.B.size(2), self.B.size(3))

        # 更新卷积层的权重
        original_weight = self.conv_layer.weight + low_rank_weight
        original_bias = self.conv_layer.bias

        # 应用卷积
        output = nn.functional.conv2d(x, original_weight, original_bias, stride=self.conv_layer.stride,
                                      padding=self.conv_layer.padding)

        return output

class CLIPWithLoRA(nn.Module):
    def __init__(self, clip_model, layers_to_replace, embedding_dim, projection_dim, num_classes, dropout=0.2):
        super(CLIPWithLoRA, self).__init__()
        self.clip_model = clip_model

        if (args.modeltype != 'B16') & (args.modeltype != 'B32'):
            self.lora_layers = []
            for layer_name, block_idx in layers_to_replace:
                layer = getattr(self.clip_model, layer_name)[block_idx]

                layer.conv1 = LoRAConv2d(layer.conv1)
                layer.conv2 = LoRAConv2d(layer.conv2)
                layer.conv3 = LoRAConv2d(layer.conv3)

                getattr(self.clip_model, layer_name)[block_idx] = layer

                self.lora_layers.extend([layer.conv1, layer.conv2, layer.conv3])

            # self.clip_model.attnpool.c_proj = LoRALayer_vit(self.clip_model.attnpool.c_proj)

        else:
            self.lora_layers = nn.ModuleList([LoRALayer_vit(layer) for layer in layers_to_replace])
            for i in range(len(layers_to_replace)):
                self.clip_model.transformer.resblocks[i].attn = self.lora_layers[i]

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.fc = nn.Linear(projection_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images):
        outputs = self.clip_model(images)
        x = outputs.flatten(start_dim=1)
        x = self.projection(x)
        x = self.fc(x)
        return x

def train_epoch(model, train_loader, optimizer, criterion, lr_scheduler, metric, accuracy_metric, precision_metric, recall_metric, f1_score_metric):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for image, label in tqdm_object:
        image, label = image.to(device), label.to(device)
        prediction = model(image)
        loss = criterion(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_meter.update(loss.item())

        metric.update(prediction, label)  # Existing
        accuracy_metric.update(prediction, label)  # Existing
        precision_metric.update(prediction, label)  # Existing
        f1_score_metric.update(prediction, label)  # Added
        recall_metric.update(prediction, label)  # Added

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    auroc = metric.compute()  # Existing
    acc = accuracy_metric.compute()  # Existing
    prec = precision_metric.compute()  # Existing
    f1_score = f1_score_metric.compute()  # Added
    recall = recall_metric.compute()  # Added

    metric.reset()  # Existing
    accuracy_metric.reset()  # Existing
    precision_metric.reset()  # Existing
    f1_score_metric.reset()  # Added
    recall_metric.reset()  # Added

    wandb.log({'train_acc': acc.item(),
               'train_loss': loss_meter.avg})

    return loss_meter, {'AUROC': auroc.item(), 'train acc': acc.item(), 'train pre': prec.item(),
                        'train f1 score': f1_score.item(), 'train recall': recall.item(), 'loss': loss_meter.avg}

def valid_epoch(model, valid_loader, criterion, metric, accuracy_metric, precision_metric, recall_metric, f1_score_metric, mode='val'):
    loss_meter = AvgMeter()
    predictions_list = []  # 初始化一个列表来收集预测结果

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))

    confusion_matrix_metric = torchmetrics.ConfusionMatrix(num_classes=len(diagnosis_map), task='multiclass').to(device)

    for image, label in tqdm_object:
        image, label = image.to(device), label.to(device)
        prediction = model(image)
        loss = criterion(prediction, label)

        metric.update(prediction, label)  # Existing
        accuracy_metric.update(prediction, label)  # Existing
        precision_metric.update(prediction, label)  # Existing
        f1_score_metric.update(prediction, label)  # Added
        recall_metric.update(prediction, label)  # Added

        if mode == 'test':
            confusion_matrix_metric.update(prediction.argmax(dim=1), label)
            predictions_list.append(prediction.detach().cpu().numpy())  # 收集预测结果

        loss_meter.update(loss.item())
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer), Recall=val_metric)

    val_metric_result = metric.compute()  # Existing
    acc = accuracy_metric.compute()  # Existing
    prec = precision_metric.compute()  # Existing
    f1_score = f1_score_metric.compute()  # Added
    recall = recall_metric.compute()  # Added

    # Reset all metrics
    metric.reset()  # Existing
    accuracy_metric.reset()  # Existing
    precision_metric.reset()  # Existing
    f1_score_metric.reset()  # Added
    recall_metric.reset()  # Added

    ##if mode is test then add it file
    if mode == 'test':
        predictions_array = np.concatenate(predictions_list, axis=0)
        np.save(npy_name, predictions_array)  # 保存文件，命名区分模式

        path_for_saving_logs = os.path.join(args.path, cm_name)
        confusion_matrix = confusion_matrix_metric.compute()
        with open(path_for_saving_logs, 'a') as f:
            f.write('Confusion Matrix:\n')
            f.write(str(confusion_matrix.cpu().numpy()) + '\n')
        confusion_matrix_metric.reset()

        wandb.log({f'{mode}_acc': acc.item(),
                   f'{mode}_loss': loss_meter.avg})

        print(loss_meter, 'testidation AUROC', val_metric_result.item(), 'test acc', acc.item(), 'test pre',
              prec.item(),
              'test f1 score', f1_score.item(), 'test recall', recall.item(), 'loss', loss_meter.avg)

        return loss_meter, {'Test AUROC': val_metric_result.item(), 'test acc': acc.item(), 'test pre': prec.item(),
                            'test f1 score': f1_score.item(), 'test recall': recall.item(), 'loss': loss_meter.avg}

    else:
        wandb.log({f'{mode}_acc': acc.item(),
                   f'{mode}_loss': loss_meter.avg})
        print(loss_meter, 'Validation AUROC', val_metric_result.item(), 'val acc', acc.item(), 'val pre', prec.item(),
              'val f1 score', f1_score.item(), 'val recall', recall.item(), 'loss', loss_meter.avg)

        return loss_meter, {'Validation AUROC': val_metric_result.item(), 'val acc': acc.item(), 'val pre': prec.item(),
                            'val f1 score': f1_score.item(), 'val recall': recall.item(), 'loss': loss_meter.avg}

def configure_model_and_transform(args, diagnosis_map, device):
    model_type_map = {
        'B32': 'ViT-B/32',
        'B16': 'ViT-B/16',
        'default': 'RN50x4'
    }
    model_name = model_type_map.get(args.modeltype, 'RN50x4')

    model, val_transform = clip.load(model_name, device=device)
    train_transform = create_train_transform(val_transform)
    model = model.visual

    lr = args.lr

    if args.mode != 'ft':
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    if args.mode == 'lora':
        layers_to_replace = (
            [(layer_name, block_idx)
             for layer_name, num_block in zip(["layer1", "layer2", "layer3", "layer4"], [0, 0, 10, 6])
             for block_idx in range(num_block)]
            if model_name.startswith('RN') else
            [model.transformer.resblocks[i].attn for i in range(12)]
        )
        embedding_dim = 640 if model_name.startswith('RN') else 512
        model = CLIPWithLoRA(model, layers_to_replace, embedding_dim, 1024, len(diagnosis_map))
    else:
        embedding_dim = 512 if 'ViT' in model_name else 640
        model = MLPHead(model, embedding_dim, 1024, len(diagnosis_map)) \
            if args.mode == 'mlp' else ProjectionHead(model, embedding_dim, 1024, len(diagnosis_map))

    model = model.float()
    model.to(device)

    return model, val_transform, train_transform, lr

if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Learning Probing on CLIP!')
    parser.add_argument('--seed', default=77, type=int, help='seed value')
    parser.add_argument('--batchsize', default=32, type=int, help='Linear probing Batch size')
    parser.add_argument('--pretrain_epoch', default=25, type=int, help='Linear probing epochs')
    parser.add_argument('--path', default='/mnt/disk/sxyy/code_proj/codecopy/LoRA/logs', type=str)
    parser.add_argument('--modeltype', default='B16', type=str, help='B32， B16， RN50x4')
    parser.add_argument('--epochs', default=20, type=int, help='Epochs')
    parser.add_argument('--mode', default='lp', type=str, help='lp / mlp / ft / lora ')
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate ')
    parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')

    args = parser.parse_args()

    seed_everything(args.seed)

    name_message = f"{args.mode}_Clip_{args.modeltype}_bs{args.batchsize}_epochs{args.epochs}_lr{args.lr}_{gettime_str()}"

    log_name = f"{name_message}_logs.txt"
    pth_name = f"{name_message}.pth"
    cm_name = f"{name_message}_cm.txt"
    npy_name = f"{name_message}.npy"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Device:' + device)

    # Fetch environment variables
    img_filepath = '/mnt/disk/sxyy/code_proj/trash_class'
    csv_path_all = '/mnt/disk/sxyy/code_proj/trash_class/csv/all.csv'
    csv_path_train = '/mnt/disk/sxyy/code_proj/trash_class/csv/train.csv'
    csv_path_test = '/mnt/disk/sxyy/code_proj/trash_class/csv/test.csv'
    csv_path_val = '/mnt/disk/sxyy/code_proj/trash_class/csv/val.csv'

    def get_categories_list(csv_file_path: str) -> List[str]:
        """
        Reads the header of a CSV file and converts each column name into a string.

        Args:
            csv_file_path (str): Path to the CSV file.

        Returns:
            List[str]: List of column names from the CSV file excluding the 'path' column.
        """
        df = pd.read_csv(csv_file_path, nrows=0)
        columns = df.columns.tolist()[1:]  # 排除 'path' 列
        return columns

    print("=> The file is from this directory:", os.getcwd())
    print("=> current running file: CLIP_linear_probe.py")
    print("=> pretrained model path:", args.path)
    print("=> batch size:", args.batchsize)
    print("=> seed:", args.seed)

    categories = get_categories_list(csv_path_all)
    diagnosis_map = {category: idx for idx, category in enumerate(categories)}
    print("Diagnosis Map:", diagnosis_map)

    # Configure model and transforms
    model, val_transform, train_transform, lr = configure_model_and_transform(args, diagnosis_map, device)
    print_trainable_parameters(model)
    get_weight_decay_params(model)

    # Create datasets
    train_dataset = Dataset(train_transform, csv_path_train, 'train', diagnosis_map)
    val_dataset = Dataset(val_transform, csv_path_val, 'val', diagnosis_map)
    test_dataset = Dataset(val_transform, csv_path_test, 'test', diagnosis_map)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4,
                                              pin_memory=True)

    # Initialize Weights & Biases
    wandb_id = os.path.split(args.path)[-1]
    wandb.init(project=name_message, id=name_message, config=args, resume='allow')

    # Define optimizer, loss, scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0, last_epoch=-1)

    # Define metrics
    train_metric = AUROC(task="multiclass", num_classes=len(diagnosis_map), average="macro", thresholds=None).to(device)
    val_metric = AUROC(task="multiclass", num_classes=len(diagnosis_map), average="macro", thresholds=None).to(device)
    test_metric = AUROC(task="multiclass", num_classes=len(diagnosis_map), average="macro", thresholds=None).to(device)

    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(device)
    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(device)
    f1_score_metric = torchmetrics.F1Score(task="multiclass", num_classes=len(diagnosis_map), average='macro').to(device)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss, train_report = train_epoch(model, train_loader, optimizer, criterion, lr_scheduler, train_metric,
                                               accuracy_metric, precision_metric, recall_metric, f1_score_metric)
        model.eval()
        with torch.no_grad():
            valid_loss, validation_report = valid_epoch(model, valid_loader, criterion, val_metric, accuracy_metric,
                                                        precision_metric, recall_metric, f1_score_metric)
            print("args.path:", args.path)
            if not os.path.exists(args.path):
                os.makedirs(args.path)
            path_for_saving_logs = os.path.join(args.path, log_name)
            print("path_for_saving_logs:", path_for_saving_logs)
            with open(path_for_saving_logs, 'a') as f:
                print("writing data at:", path_for_saving_logs)
                f.write(json.dumps(train_report) + '\n')
                f.write(json.dumps(validation_report) + '\n')
            print("=> done with writing::")
            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                print("The best loss so far:", best_loss)
                torch.save(model.state_dict(), pth_name)

    # Load the best model and evaluate on test set
    model.load_state_dict(torch.load(pth_name))
    model.eval()
    with torch.no_grad():
        test_loss, test_report = valid_epoch(model, test_loader, criterion, test_metric, accuracy_metric,
                                             precision_metric, recall_metric, f1_score_metric, mode='test')
        print("The final test report is:", test_report)
        with open(path_for_saving_logs, 'a') as f:
            f.write(json.dumps(test_report) + '\n')
