"""
CARLA 0.9.8 模仿学习训练脚本（适配Python3.7 + torchvision0.12.0 + cu116）
功能：加载CARLA划分的训练集/验证集，训练ResNet34控制模型（集成ImageNet预训练权重）
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD, Adam
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pickle
import os
import sys
from collections import OrderedDict  # Python3.7必备
from multiprocessing import freeze_support  # Windows多进程支持
import matplotlib.pyplot as plt

# -------------------------- 版本兼容性检查 --------------------------
def print_env_info():
    print(f"Python版本: {sys.version[:3]}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"TorchVision版本: {torchvision.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")

# -------------------------- 配置参数 --------------------------
CONFIG = {
    "TRAIN_DATA_PATH": "D:/Project/reinforence_learning/g-ajwe4360-carla_drl-v3.0-/training_carla.pkl",
    "VAL_DATA_PATH": "D:/Project/reinforence_learning/g-ajwe4360-carla_drl-v3.0-/validation_carla.pkl",
    "DEVICE_ID": 0,
    "TRAIN_BATCH_SIZE": 50,
    "VAL_BATCH_SIZE": 50,
    "EPOCHS": 120,
    "INIT_LR": 0.0001,
    "FINETUNE_LR": 0.00001,  # 解冻后的学习率
    "WEIGHT_DECAY": 1e-4,
    "PRINT_EVERY": 1,
    "EVAL_EVERY": 1,
    "SAVE_EVERY": 10,
    "LOG_DIR": "RL_ResNet34_CARLA_Pretrained",
    "CHECKPOINT_PREFIX": "RL_ResNet34_CARLA_Pretrained-",
    "TRAIN_MEAN": (0.1840, 0.1659, 0.1613),
    "TRAIN_STD": (0.2540, 0.2386, 0.2599),
    "NUM_WORKERS": 0,  # Python3.7+Windows下强制0
    "FINETUNE_EPOCH": 60,  # 第60个epoch开始解冻微调
}

# -------------------------- 模型定义（适配torchvision0.12.0） --------------------------
class ResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet34, self).__init__()
        # 使用torchvision0.12.0的ResNet34，加载ImageNet预训练权重（适配旧版本的pretrained参数）
        # torchvision0.12.0中，pretrained=True 对应加载ImageNet1K预训练权重
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        self.resnet.fc = nn.Identity()  # 移除原全连接层，保留512维特征

        # 运动状态分支 - 重命名为kinematics_encoder，与DDPG Actor保持一致
        self.kinematics_fc = nn.Sequential(
            nn.Linear(2, 32),# 关键修改：in_features从1→2
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        # 融合全连接层 - 重命名为fusion_encoder，与DDPG Actor保持一致
        self.fc1 = nn.Linear(512 + 32, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        # 动作输出层 - 重命名为action_head，与DDPG Actor保持一致
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x, kinematics):
        """
        前向传播函数
        Args:
            x: 图像输入，shape [batch_size, 3, H, W]
            kinematics: 运动状态输入，shape [batch_size, 1]
        Returns:
            control: 控制输出，shape [batch_size, 3]，其中：
                     - 前两列：油门/刹车，范围0~1
                     - 第三列：方向盘，范围-1~1
        """
        # 图像特征提取
        img_feat = self.resnet(x)  # shape: [batch_size, 512]

        # 运动状态特征提取
        kin_feat = self.kinematics_fc(kinematics)  # shape: [batch_size, 32]

        # 特征融合
        fusion_feat = torch.cat([img_feat, kin_feat], dim=1)  # shape: [batch_size, 544]
        fusion_feat = self.relu(self.fc1_bn(self.fc1(fusion_feat)))  # shape: [batch_size, 256]
        fusion_feat = self.relu(self.fc2_bn(self.fc2(fusion_feat)))  # shape: [batch_size, 128]

        # 动作输出
        control = self.fc3(fusion_feat)  # shape: [batch_size, 3]

        # 激活函数处理
        # 油门/刹车：sigmoid激活，范围0~1
        # 方向盘：tanh激活，范围-1~1
        throttle_brake = torch.sigmoid(control[:, :2])  # shape: [batch_size, 2]
        steering = torch.tanh(control[:, 2:3])  # shape: [batch_size, 1]

        # 拼接完整控制输出
        control = torch.cat([throttle_brake, steering], dim=1)  # shape: [batch_size, 3]

        return control

# 兼容原代码的空实现（同样添加预训练权重支持）
class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Identity()
        self.kinematics_fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.fc1 = nn.Linear(2048 + 32, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, x, kinematics):
        img_feat = self.resnet(x)
        kin_feat = self.kinematics_fc(kinematics)
        fusion_feat = torch.cat([img_feat, kin_feat], dim=1)
        x = self.relu(self.fc1_bn(self.fc1(fusion_feat)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        control = self.fc3(x)
        control = torch.sigmoid(control)
        control[:, 2] = (control[:, 2] - 0.5) * 2
        return control

class SE_ResNet34(ResNet34):
    def __init__(self, pretrained=True):
        super(SE_ResNet34, self).__init__(pretrained=pretrained)

class BAM_ResNet34(ResNet34):
    def __init__(self, pretrained=True):
        super(BAM_ResNet34, self).__init__(pretrained=pretrained)

class CBAM_ResNet34(ResNet34):
    def __init__(self, pretrained=True):
        super(CBAM_ResNet34, self).__init__(pretrained=pretrained)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
    def forward(self, x):
        return x

# -------------------------- 加载数据集 --------------------------
def load_carla_data(file_path: str) -> list:
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        raise FileNotFoundError(f"未找到数据集：{file_path}")
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding='latin1')  # Python3.7兼容
        print(f"成功加载 {file_path}：{len(data)} 条数据")
        return data
    except Exception as e:
        raise RuntimeError(f"加载失败：{e}")

# -------------------------- 数据集类 --------------------------
class CarlaTrainingDataset(Dataset):
    def __init__(self, data: list, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 解包8元素数据
        prev_img, prev_nav, continuous_action, discrete_action, reward, curr_img, curr_nav, done = self.data[index]

        # 对于模仿学习，我们只需要当前图像和连续动作
        image = curr_img

        # 提取速度（修正：添加单位转换，从km/h转为m/s）
        velocity_kmh = prev_nav[1]
        velocity = velocity_kmh / 3.6  # 转换为m/s，匹配物理单位
        waypoint_dist = prev_nav[5]  # 假设prev_nav[4]是到下一个弯道的距离（关键特征！），需根据你的数据结构调整

        # 组合多维度运动状态特征（原代码是单维度速度，现在扩展为2维度）
        kinematics = np.array([velocity, waypoint_dist], dtype=np.float32)

        if self.transform is not None:
            from PIL import Image
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
            velocity = torch.tensor(velocity, dtype=torch.float32).unsqueeze(0)

        # 提取动作（转向、油门、刹车）
        steering = continuous_action[0]
        throttle = continuous_action[1]
        brake = continuous_action[2]

        # 控制动作：使用连续动作
        control = torch.tensor([throttle, brake, steering], dtype=torch.float32)

        return image, kinematics, control

class CarlaValDataset(Dataset):
    def __init__(self, data: list, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 与训练集相同的处理逻辑
        prev_img, prev_nav, continuous_action, discrete_action, reward, curr_img, curr_nav, done = self.data[index]

        image = curr_img

        # 提取速度（修正：添加单位转换）
        velocity_kmh = prev_nav[1]
        velocity = velocity_kmh / 3.6
        waypoint_dist = prev_nav[5]  # 同样调整为你的弯道距离索引

        # 组合多维度运动状态特征
        kinematics = np.array([velocity, waypoint_dist], dtype=np.float32)

        if self.transform is not None:
            from PIL import Image
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
            velocity = torch.tensor(velocity, dtype=torch.float32).unsqueeze(0)

        # 提取动作
        steering = continuous_action[0]
        throttle = continuous_action[1]
        brake = continuous_action[2]

        # 控制动作：使用连续动作
        control = torch.tensor([throttle, brake, steering], dtype=torch.float32)

        return image, kinematics, control

# -------------------------- 层冻结/解冻 --------------------------
def freeze_layer(layer):
    """冻结指定层的参数"""
    for param in layer.parameters():
        param.requires_grad = False
        param.grad = None

def unfreeze_layer(layer):
    """解冻指定层的参数"""
    for param in layer.parameters():
        param.requires_grad = True


# -------------------------- 转向时机约束损失函数（新增） --------------------------
def steering_timing_loss(pred_steer, true_steer, waypoint_dist, alpha=1.0):
    """
    转向时机约束损失：惩罚距离弯道较远时的过大转向角（解决提前左转问题）
    Args:
        pred_steer: 预测转向角度, shape [batch_size, 1]
        true_steer: 专家转向角度, shape [batch_size, 1]
        waypoint_dist: 到下一个弯道的距离, shape [batch_size, 1]（单位：m）
        alpha: 权重系数，控制约束强度
    Returns:
        timing_loss: 时机约束损失（标量）
    """
    # 距离越远，权重越大，强制转向角度接近0（指数衰减权重，可根据场景调整）
    weight = torch.exp(-waypoint_dist / 50.0)  # 距离100m时权重≈0.37，50m时≈0.61，0m时≈1
    # 两部分损失：1. 预测转向角与0的差（惩罚过早转向）；2. 预测与专家的回归差
    timing_penalty = weight * torch.square(pred_steer)
    reg_loss = torch.square(pred_steer - true_steer)
    # 总时机损失（取均值保证尺度稳定）
    return alpha * torch.mean(timing_penalty + reg_loss)

# -------------------------- 学习率调度器 --------------------------
def lr_scheduler(lr, epoch, finetune_epoch=CONFIG["FINETUNE_EPOCH"], finetune_lr=CONFIG["FINETUNE_LR"]):
    """
    学习率调度：前finetune_epoch个epoch使用初始学习率，之后使用微调学习率，且每30个epoch衰减
    """
    if epoch < finetune_epoch:
        # 冻结阶段：初始学习率衰减
        return lr * (0.3 ** (epoch // 30))
    else:
        # 解冻阶段：微调学习率衰减
        return finetune_lr * (0.3 ** ((epoch - finetune_epoch) // 30))

# -------------------------- 主训练函数 --------------------------
def main():
    print_env_info()
    # 设备配置
    device = torch.device(
        f"cuda:{CONFIG['DEVICE_ID']}" if (torch.cuda.is_available() and torch.version.cuda.startswith("11.6")) else "cpu"
    )
    print(f"\n使用设备：{device}")
    if device.type == "cuda":
        torch.cuda.set_device(CONFIG['DEVICE_ID'])
        torch.backends.cudnn.benchmark = True

    # 加载数据集
    training_data = load_carla_data(CONFIG["TRAIN_DATA_PATH"])
    val_data = load_carla_data(CONFIG["VAL_DATA_PATH"])

    # 数据预处理（添加训练集数据增强，验证集不增强）
    transform_train = transforms.Compose([
        transforms.Resize((480, 640)),
        # 数据增强：随机水平翻转（自动驾驶需注意，左转弯场景可注释，通用场景保留）
        # transforms.RandomHorizontalFlip(p=0.5),
        # 数据增强：随机亮度调整
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["TRAIN_MEAN"], CONFIG["TRAIN_STD"])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["TRAIN_MEAN"], CONFIG["TRAIN_STD"])
    ])

    # 数据加载器
    train_set = CarlaTrainingDataset(data=training_data, transform=transform_train)
    val_set = CarlaValDataset(data=val_data, transform=transform_val)
    train_loader = DataLoader(
        train_set,
        batch_size=CONFIG["TRAIN_BATCH_SIZE"],
        shuffle=True,
        num_workers=CONFIG["NUM_WORKERS"],
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=CONFIG["VAL_BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"],
        pin_memory=True if device.type == "cuda" else False,
        drop_last=False
    )
    print(f"训练集批次：{len(train_loader)} | 验证集批次：{len(val_loader)}")

    # 模型初始化（加载ImageNet预训练权重）
    net = ResNet34(pretrained=True).to(device)
    print("\n模型实例化完成：ResNet34（加载ImageNet预训练权重）")

    # 初始冻结策略：冻结ResNet34骨干网络，只训练自定义分支
    freeze_layer(net.resnet)
    unfreeze_layer(net.kinematics_fc)
    unfreeze_layer(net.fc1)
    unfreeze_layer(net.fc1_bn)
    unfreeze_layer(net.fc2)
    unfreeze_layer(net.fc2_bn)
    unfreeze_layer(net.fc3)
    print("\n初始层状态：冻结ResNet34骨干网络，解冻自定义分支")

    # 损失函数和优化器（使用Adam优化器，收敛更快；也可保留SGD）
    criterion = nn.SmoothL1Loss()
    optimizer = Adam(  # 替换SGD为Adam，提升收敛速度
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=CONFIG["INIT_LR"],
        weight_decay=CONFIG["WEIGHT_DECAY"]
    )
    # 若保留SGD：
    # optimizer = SGD(
    #     filter(lambda p: p.requires_grad, net.parameters()),
    #     lr=CONFIG["INIT_LR"],
    #     momentum=0.9,
    #     weight_decay=CONFIG["WEIGHT_DECAY"],
    #     nesterov=True
    # )
    print("损失函数和优化器初始化完成")

    # 训练准备
    train_losses = []
    eval_losses = []
    writer = SummaryWriter(log_dir=CONFIG["LOG_DIR"])
    global_step = 0
    finetune_flag = False  # 解冻标志

    # 训练循环
    print("\n开始训练...")
    for e in range(1, CONFIG["EPOCHS"] + 1):
        train_loss = 0.0
        count = 0
        # 更新学习率
        lr_cur = lr_scheduler(CONFIG["INIT_LR"], e)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_cur
        print(f"\n{'='*50}\nEpoch: {e}/{CONFIG['EPOCHS']} | 学习率: {lr_cur:.7f}\n{'='*50}")
        writer.add_scalar("training/learning_rate", lr_cur, global_step)

        # 解冻策略：第FINETUNE_EPOCH个epoch解冻ResNet34骨干网络
        if e == CONFIG["FINETUNE_EPOCH"] and not finetune_flag:
            unfreeze_layer(net.resnet)
            # 可选：只解冻ResNet34的后几层，保留前几层的预训练特征
            # for name, param in net.resnet.named_parameters():
            #     if "layer4" in name or "layer3" in name:
            #         param.requires_grad = True
            #     else:
            #         param.requires_grad = False
            print(f"\nEpoch {e}：解冻ResNet34骨干网络，开始微调")
            finetune_flag = True

        # 训练阶段
        net.train()
        for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            imgs, kinematics_states, controls = data
            imgs = imgs.to(device, non_blocking=True)
            kinematics_states = kinematics_states.to(device, non_blocking=True)
            controls = controls.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # 前向传播
            out = net(imgs.float(), kinematics_states.float())
            # 计算损失（保留*2缩放，加速收敛）
            reg_loss = criterion(out * 2, controls.float() * 2)

            # 2. 转向时机约束损失（新增）
            pred_steer = out[:, 2:3]  # 预测转向角，shape: [batch, 1]
            true_steer = controls[:, 2:3]  # 专家转向角，shape: [batch, 1]
            waypoint_dist = kinematics_states[:, 1:2]  # 弯道距离（kinematics_states[:,1]是弯道距离，[:,0]是速度）
            timing_loss = steering_timing_loss(pred_steer, true_steer, waypoint_dist, alpha=1.0)  # alpha可调整

            # 3. 复合总损失（权重可根据效果调整）
            total_loss = reg_loss + timing_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            count += 1
            global_step += 1
            writer.add_scalar("training/huber_loss", reg_loss.item(), global_step)
            writer.add_scalar("training/timing_loss", timing_loss.item(), global_step)
            writer.add_scalar("training/total_loss", total_loss.item(), global_step)
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Batch {batch_idx + 1}/{len(train_loader)} | Reg Loss: {reg_loss.item():.6f} | Timing Loss: {timing_loss.item():.6f} | Total Loss: {total_loss.item():.6f}")

        # 训练损失统计
        avg_train_loss = train_loss / count
        train_losses.append(avg_train_loss)
        print(f"\nEpoch {e} 训练完成 | 平均Loss: {avg_train_loss:.6f}")

        # 验证阶段
        if e % CONFIG["EVAL_EVERY"] == 0:
            net.eval()
            eval_loss = 0.0
            val_count = 0
            predicted_steerings = []
            true_steerings = []
            print(f"\n{'='*30} Epoch {e} 验证阶段 {'='*30}")
            with torch.no_grad():
                for batch_idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                    imgs, kinematics_states, controls = data
                    imgs = imgs.to(device, non_blocking=True)
                    kinematics_states = kinematics_states.to(device, non_blocking=True)
                    controls = controls.to(device, non_blocking=True)
                    out = net(imgs.float(), kinematics_states.float())
                    loss_val = criterion(out * 2, controls.float() * 2)
                    eval_loss += loss_val.item()
                    val_count += 1

                    # 收集预测和真实的转向角度
                    predicted_steerings.extend(out[:, 2].cpu().numpy())
                    true_steerings.extend(controls[:, 2].cpu().numpy())

            avg_eval_loss = eval_loss / val_count
            eval_losses.append(avg_eval_loss)
            writer.add_scalar("val/huber_loss", avg_eval_loss, global_step)
            print(f"Epoch {e} 验证完成 | 平均Loss: {avg_eval_loss:.6f}")

            # 分析转向分布
            print(f"\n模型预测转向分布：")
            print(f"预测 - 均值: {np.mean(predicted_steerings):.4f}, 标准差: {np.std(predicted_steerings):.4f}")
            print(f"真实 - 均值: {np.mean(true_steerings):.4f}, 标准差: {np.std(true_steerings):.4f}")

            # 可视化预测vs真实转向（可选开启）
            # plt.figure(figsize=(10, 5))
            # plt.subplot(121)
            # plt.hist(predicted_steerings, bins=50, alpha=0.7, color='blue', label='Predicted')
            # plt.hist(true_steerings, bins=50, alpha=0.7, color='orange', label='True')
            # plt.title(f'Epoch {e} Steering Distribution')
            # plt.xlabel('Steering Angle')
            # plt.ylabel('Count')
            # plt.legend()
            # plt.subplot(122)
            # plt.scatter(true_steerings, predicted_steerings, alpha=0.5, s=1)
            # plt.plot([-1, 1], [-1, 1], 'r--')  # 理想线
            # plt.title(f'True vs Predicted Steering')
            # plt.xlabel('True Steering')
            # plt.ylabel('Predicted Steering')
            # plt.xlim([-1, 1])
            # plt.ylim([-1, 1])
            # plt.tight_layout()
            # plt.savefig(f'epoch_{e}_steering_prediction.png')
            # plt.close()

        # 保存模型
        if e % CONFIG["SAVE_EVERY"] == 0:
            checkpoint_path = f"{CONFIG['CHECKPOINT_PREFIX']}epoch{e}-loss{avg_train_loss:.4f}.pth"
            torch.save({
                'epoch': e,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),  # 保存优化器状态
                'train_loss': avg_train_loss,
                'val_loss': avg_eval_loss if eval_losses else None
            }, checkpoint_path)
            print(f"\n模型已保存：{checkpoint_path}")

        # 累计损失统计
        print(f"\n截至Epoch {e} | 累计平均训练Loss: {sum(train_losses)/e:.6f}")

    # 训练结束
    writer.close()
    print("\n" + "="*50)
    print("训练完成！")
    print(f"日志目录：{CONFIG['LOG_DIR']}")
    print(f"最后训练Loss：{train_losses[-1]:.6f}")
    if eval_losses:
        print(f"最后验证Loss：{eval_losses[-1]:.6f}")
    print("="*50)

# -------------------------- 主入口（Python3.7+Windows必须） --------------------------
if __name__ == '__main__':
    freeze_support()
    main()