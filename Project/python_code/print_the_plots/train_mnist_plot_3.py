import argparse
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import io

import torch
import torch.nn as nn
import torchvision.datasets
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from munkres import Munkres
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'channel_parameters': {
        'user_max_power': 1.0,
        'uplink_bandwidth': 1e6,
        'downlink_bandwidth': 1e6,
        'psi': 1e-27,
        'omega_i': 40 * 1e18,
        'theta': 1.0,
        'delay_requirement': 0.5,
        'energy_requirement': 0.003,
    },
    'model_hyperparameters': {
        'num_channels': 1,
    }
}

def simple_logger(message):
    print(message)


class NeuralNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def packet_error_calculator(distance_matrix, channel_interference, user_max_power):
    epsilon = 1e-6
    distance_matrix = np.maximum(distance_matrix, epsilon)
    q_packet_error = 1 - np.exp(-1.08 * (channel_interference + 1e-14) /
                                (user_max_power * np.power(distance_matrix, -2)))
    return q_packet_error

def sinr_calculator(distance_matrix, channel_interference_matrix, power=1):
    epsilon = 1e-6
    distance_matrix = np.maximum(distance_matrix, epsilon)
    SINR = power * np.divide(np.power(distance_matrix, -2),
                             channel_interference_matrix + epsilon)
    return SINR

def per_user_total_energy_calculator(fl_model_data_size, uplink_delay, psi, omega_i, theta, user_power):
    total_energy = psi * omega_i * (theta ** 2) * fl_model_data_size + user_power * uplink_delay
    return total_energy

def channel_rate_calculator(bandwidth, sinr):
    sinr = np.clip(sinr, 1e-6, 1e6)
    rate = bandwidth * np.log2(1 + sinr)
    return rate

def evaluate(model, test_loader, writer, criterion, iteration, strategy):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    device = next(model.parameters()).device

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    average_loss = total_loss / total

    simple_logger(f'[{strategy}] Iter {iteration}: Test Acc: {test_accuracy:.2f}%, Loss: {average_loss:.4f}')
    writer.add_scalar(f"{strategy}/accuracy/test_global_model", test_accuracy, global_step=iteration)
    writer.add_scalar(f"{strategy}/loss/test_global_model", average_loss, global_step=iteration)

    model.train()
    return average_loss, test_accuracy


def run_fl_for_user_blocks(args, user_blocks, writer, strategy_list):
    simple_logger(f"\n===== Starting FL for user_blocks = {user_blocks} =====")

    mnkr = Munkres()

    user_number = args.user_number
    datanumber = [100] * user_number

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

    if user_blocks == 3:
        channel_interference = (np.array([0.05, 0.1, 0.14]) - 0.04) * 0.000001
    elif user_blocks == 6:
        channel_interference = (np.array([0.05, 0.07, 0.09, 0.11, 0.13, 0.15]) - 0.04) * 0.000001
    elif user_blocks == 9:
        channel_interference = (np.array([0.03, 0.06, 0.07, 0.08, 0.1, 0.11, 0.12, 0.14, 0.15]) - 0.04) * 0.000001
    elif user_blocks == 12:
        channel_interference = (np.array([0.03, 0.05, 0.06, 0.07, 0.08, 0.09,
                                          0.1, 0.11, 0.12, 0.13, 0.14, 0.15]) - 0.04) * 0.000001
    else:
        raise ValueError("Unsupported user_blocks value.")

    channel_interference_downlink = 0.06 * 0.000003

    distance = np.random.rand(user_number, 1) * 500
    q_packet_error = packet_error_calculator(distance, channel_interference, config['channel_parameters']['user_max_power'])
    sinr = sinr_calculator(distance, channel_interference, config["channel_parameters"]["user_max_power"])
    rateu = channel_rate_calculator(config['channel_parameters']["uplink_bandwidth"], sinr)
    SINRd = sinr_calculator(distance, channel_interference_downlink)
    rated = channel_rate_calculator(config['channel_parameters']['downlink_bandwidth'], SINRd)

    models_dict = {}
    for user in range(user_number):
        local_model = NeuralNet().to(args.device)
        local_model.apply(init_weights)
        models_dict[f"model_{user}"] = local_model
    
    total_model_params = sum(p.numel() for p in models_dict['model_0'].parameters())
    Z = total_model_params * 16 / 1024 / 1024

    delayu = Z / rateu
    delayd = Z / rated
    totaldelay = delayu + delayd
    totalenergy = per_user_total_energy_calculator(
        Z,
        delayu,
        config['channel_parameters']['psi'],
        config['channel_parameters']['omega_i'],
        config['channel_parameters']['theta'],
        config["channel_parameters"]["user_max_power"],
    )

    final_accuracies = {}

    for strategy in strategy_list:
        simple_logger(f"--- Strategy: {strategy} ---")

        # models_dict = {}
        # for user in range(user_number):
        #     local_model = NeuralNet().to(args.device)
        #     local_model.apply(init_weights)
        #     models_dict[f"model_{user}"] = local_model

        num_resource_blocks = len(channel_interference)
        W = np.zeros((user_number, num_resource_blocks))
        finalq = np.ones((1, user_number))

        if strategy == 'proposed':
            for i in range(user_number):
                for j in range(num_resource_blocks):
                    if (totaldelay[i, j] < config['channel_parameters']['delay_requirement']) and \
                       (totalenergy[i, j] < config['channel_parameters']['energy_requirement']):
                        W[i, j] = -datanumber[i] * (1 - q_packet_error[i, j])
                    else:
                        W[i, j] = 1e+10
            try:
                assignment = mnkr.compute(W.tolist())
            except Exception as e:
                simple_logger(f"[proposed] Hungarian failed: {e}")
                assignment = []
            for assign in assignment:
                if len(assign) == 2:
                    i, j = assign
                    if W[i][j] != 1e+10:
                        finalq[0, i] = q_packet_error[i, j]
                    else:
                        finalq[0, i] = 1

        elif strategy in ['baseline1', 'baseline3']:
            qassignment = np.zeros((1, user_number))
            assignment = np.zeros((1, user_number))

            if num_resource_blocks < user_number:
                selected_users = random.sample(range(user_number), num_resource_blocks)
            else:
                selected_users = list(range(user_number))

            selected_rbs = random.sample(range(num_resource_blocks), len(selected_users))
            for idx, user_ in enumerate(selected_users):
                assignment[0, user_] = 1
                qassignment[0, user_] = selected_rbs[idx]

            finalq = np.ones((1, user_number))
            for user_ in range(user_number):
                j = int(qassignment[0, user_])
                if j < num_resource_blocks and assignment[0, user_] == 1:
                    if (totaldelay[user_, j] < config['channel_parameters']['delay_requirement']) and \
                       (totalenergy[user_, j] < config['channel_parameters']['energy_requirement']):
                        finalq[0, user_] = q_packet_error[user_, j]
                    else:
                        finalq[0, user_] = 1
                else:
                    finalq[0, user_] = 1

        elif strategy == 'baseline2':
            qassignment = np.zeros((1, user_number))
            assignment = np.zeros((1, user_number))
            if num_resource_blocks < user_number:
                selected_users = random.sample(range(user_number), num_resource_blocks)
            else:
                selected_users = list(range(user_number))
            selected_rbs = random.sample(range(num_resource_blocks), len(selected_users))
            for idx, user_ in enumerate(selected_users):
                assignment[0, user_] = 1
                qassignment[0, user_] = selected_rbs[idx]
            for user_ in range(user_number):
                j = int(qassignment[0, user_])
                if j < num_resource_blocks and assignment[0, user_] == 1:
                    if (totaldelay[user_, j] < config['channel_parameters']['delay_requirement']) and \
                       (totalenergy[user_, j] < config['channel_parameters']['energy_requirement']):
                        finalq[0, user_] = q_packet_error[user_, j]
                    else:
                        finalq[0, user_] = 1
                else:
                    finalq[0, user_] = 1

        global_weights = None
        iteration_accuracy = 0

        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        for it in tqdm(range(args.iteration), desc=f"{strategy} (RBs={user_blocks})"):
            participating_users = []
            for user_ in range(user_number):
                if (it == 0 and finalq[0, user_] != 1) or (random.random() > finalq[0, user_]):
                    participating_users.append(user_)
                    start_idx = sum(datanumber[:user_])
                    end_idx = sum(datanumber[:user_+1])
                    user_train_dataset = torch.utils.data.Subset(train_dataset, list(range(start_idx, end_idx)))
                    train_loader = DataLoader(dataset=user_train_dataset, batch_size=args.batch_size, shuffle=True)
                    optimizer = optim.Adam(models_dict[f"model_{user_}"].parameters(), lr=0.001)
                    models_dict[f"model_{user_}"].train()

                    for images, labels in train_loader:
                        images, labels = images.to(args.device), labels.to(args.device)
                        outputs = models_dict[f"model_{user_}"](images)
                        loss = criterion(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(models_dict[f"model_{user_}"].parameters(), 1.0)
                        optimizer.step()

            if participating_users:
                if global_weights is None:
                    global_weights = {}
                    for name, param in models_dict[f"model_{participating_users[0]}"].state_dict().items():
                        global_weights[name] = torch.zeros_like(param.data)

                total_data = 0
                for user_ in participating_users:
                    for name, param in models_dict[f"model_{user_}"].state_dict().items():
                        global_weights[name] += param.data * datanumber[user_]
                    total_data += datanumber[user_]

                for name in global_weights:
                    global_weights[name] /= total_data

                global_model = NeuralNet().to(args.device)
                global_model.load_state_dict(global_weights)
                _, test_acc = evaluate(global_model, test_loader, writer, criterion, it, strategy)
                iteration_accuracy = test_acc

                for user_ in range(user_number):
                    models_dict[f"model_{user_}"].load_state_dict(global_model.state_dict())

        final_accuracies[strategy] = iteration_accuracy

    return final_accuracies


def main(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    np.random.seed(0)

    writer = SummaryWriter()

    user_blocks_list = [3, 6, 9, 12]
    strategies = ['proposed', 'baseline1', 'baseline2', 'baseline3']
    results = {strategy: [] for strategy in strategies}

    for ub in tqdm(user_blocks_list, desc="UserBlocks Loop"):
        final_accs = run_fl_for_user_blocks(args, ub, writer, strategies)
        for strategy in strategies:
            results[strategy].append(final_accs[strategy])

    plt.figure(figsize=(8, 6))
    for strategy in strategies:
        plt.plot(user_blocks_list, results[strategy], marker='o', label=strategy)
    plt.xlabel("Number of RBs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Identification Accuracy vs Number of RBs (U={args.user_number})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = plt.imread(buf)
    writer.add_image('Accuracy_vs_UserBlocks', image, global_step=0, dataformats='HWC')
    buf.close()

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iteration",
        type=int,
        default=130,
        help="number of total iterations for FL training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size of the model",
    )
    parser.add_argument(
        "--user_number",
        type=int,
        default=15,
        help="Fixed total number of users",
    )
    args = parser.parse_args()

    main(args)

