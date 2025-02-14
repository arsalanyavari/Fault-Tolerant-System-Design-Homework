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

# Simple logger function
def simple_logger(message):
    print(message)

# Example configuration dictionary
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

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    # To prevent log2(1 + sinr) from overflowing, clip sinr to a reasonable range
    sinr = np.clip(sinr, 1e-6, 1e6)
    rate = bandwidth * np.log2(1 + sinr)
    return rate


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def evaluate(model, test_loader, writer, criterion, iteration, strategy):
    """Evaluate the model on the global test set."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
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


def run_fl_for_users(args, user_number, writer, strategy_list):
    """
    Runs federated learning for a given number of users (user_number) and
    returns a dictionary of final accuracies for each strategy.
    """
    simple_logger(f"\n===== Starting FL for user_number = {user_number} =====")

    mnkr = Munkres()

    datanumber = [100] * user_number

    # Downlink interference
    channel_interference_downlink = 0.06 * 0.000003

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

    # Set channel interference based on user_blocks
    if args.user_blocks == 3:
        channel_interference = (np.array([0.05, 0.1, 0.14]) - 0.04) * 0.000001
    elif args.user_blocks == 6:
        channel_interference = (np.array([0.05, 0.07, 0.09, 0.11, 0.13, 0.15]) - 0.04) * 0.000001
    elif args.user_blocks == 9:
        channel_interference = (np.array([0.03, 0.06, 0.07, 0.08, 0.1, 0.11, 0.12, 0.14, 0.15]) - 0.04) * 0.000001
    elif args.user_blocks == 12:
        channel_interference = (np.array([0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11,
                                          0.12, 0.13, 0.14, 0.15]) - 0.04) * 0.000001
    else:
        raise ValueError("Unsupported number of user blocks.")

    num_resource_blocks = len(channel_interference)

    # Generate random distances for each user
    distance = np.random.rand(user_number, 1) * 500

    # Calculate metrics
    q_packet_error = packet_error_calculator(distance, channel_interference, config['channel_parameters']['user_max_power'])
    sinr = sinr_calculator(distance, channel_interference, config["channel_parameters"]["user_max_power"])
    rateu = channel_rate_calculator(config['channel_parameters']["uplink_bandwidth"], sinr)
    SINRd = sinr_calculator(distance, channel_interference_downlink)
    rated = channel_rate_calculator(config['channel_parameters']['downlink_bandwidth'], SINRd)

    # Data size in MB (example)
    # Initialize local models
    models = {}
    for user in range(user_number):
        local_model = NeuralNet().to(device)
        local_model.apply(init_weights)
        models[f"model_{user}"] = local_model
        
    total_model_params = sum(p.numel() for p in models['model_0'].parameters())
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

    # Initialize final accuracies for each strategy
    final_accuracies = {}

    # We'll run once per strategy
    for strategy in strategy_list:
        simple_logger(f"--- Strategy: {strategy} ---")

        # # Initialize local models
        # models = {}
        # for user in range(user_number):
        #     local_model = NeuralNet().to(device)
        #     local_model.apply(init_weights)
        #     models[f"model_{user}"] = local_model

        # Resource allocation
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

        elif strategy == 'baseline1':
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
                simple_logger(f"[baseline1] Hungarian failed: {e}")
                assignment = []
            qassignment = np.zeros((1, user_number))
            assigned_users = []
            for assign in assignment:
                if len(assign) == 2:
                    i, j = assign
                    if W[i][j] != 1e+10:
                        assigned_users.append(i)
            if len(assigned_users) > 0:
                # random RB allocation
                if num_resource_blocks < len(assigned_users):
                    selected_rbs = random.sample(range(num_resource_blocks), len(assigned_users))
                else:
                    selected_rbs = random.sample(range(num_resource_blocks), len(assigned_users))
                for idx, user in enumerate(assigned_users):
                    qassignment[0, user] = selected_rbs[idx]
            for user in range(user_number):
                j = int(qassignment[0, user])
                if j < num_resource_blocks and W[user, j] != 1e+10:
                    finalq[0, user] = q_packet_error[user, j]
                else:
                    finalq[0, user] = 1

        elif strategy == 'baseline2':
            qassignment = np.zeros((1, user_number))
            assignment = np.zeros((1, user_number))
            if num_resource_blocks < user_number:
                selected_users = random.sample(range(user_number), num_resource_blocks)
            else:
                selected_users = list(range(user_number))
            selected_rbs = random.sample(range(num_resource_blocks), len(selected_users))
            for idx, user in enumerate(selected_users):
                assignment[0, user] = 1
                qassignment[0, user] = selected_rbs[idx]
            for user in range(user_number):
                j = int(qassignment[0, user])
                if j < num_resource_blocks and assignment[0, user] == 1:
                    if (totaldelay[user, j] < config['channel_parameters']['delay_requirement']) and \
                       (totalenergy[user, j] < config['channel_parameters']['energy_requirement']):
                        finalq[0, user] = q_packet_error[user, j]
                    else:
                        finalq[0, user] = 1
                else:
                    finalq[0, user] = 1

        elif strategy == 'baseline3':
            for i in range(user_number):
                for j in range(num_resource_blocks):
                    if (totaldelay[i, j] < config['channel_parameters']['delay_requirement']) and \
                       (totalenergy[i, j] < config['channel_parameters']['energy_requirement']):
                        W[i, j] = q_packet_error[i, j]
                    else:
                        W[i, j] = 1e+10
            try:
                assignment = mnkr.compute(W.tolist())
            except Exception as e:
                simple_logger(f"[baseline3] Hungarian failed: {e}")
                assignment = []
            qassignment = np.zeros((1, user_number))
            assigned_users = []
            for assign in assignment:
                if len(assign) == 2:
                    i, j = assign
                    if W[i][j] != 1e+10:
                        assigned_users.append(i)
            if len(assigned_users) > 0:
                if num_resource_blocks < len(assigned_users):
                    selected_rbs = random.sample(range(num_resource_blocks), len(assigned_users))
                else:
                    selected_rbs = random.sample(range(num_resource_blocks), len(assigned_users))
                for idx, user in enumerate(assigned_users):
                    qassignment[0, user] = selected_rbs[idx]
            for user in range(user_number):
                j = int(qassignment[0, user])
                if j < num_resource_blocks and W[user, j] != 1e+10:
                    finalq[0, user] = q_packet_error[user, j]
                else:
                    finalq[0, user] = 1

        # Training for a certain number of iterations
        global_weights = None
        iteration_accuracy = 0

        # Use tqdm for iteration loop
        for it in tqdm(range(args.iteration), desc=f"{strategy} (Users={user_number})"):
            participating_users = []
            for user in range(user_number):
                # Decide if user participates
                if (it == 0 and finalq[0, user] != 1) or (random.random() > finalq[0, user]):
                    participating_users.append(user)
                    # Local training
                    start_idx = sum(datanumber[:user])
                    end_idx = sum(datanumber[:user+1])
                    user_train_dataset = torch.utils.data.Subset(train_dataset, list(range(start_idx, end_idx)))
                    train_loader = DataLoader(dataset=user_train_dataset, batch_size=args.batch_size, shuffle=True)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(models[f"model_{user}"].parameters(), lr=0.001)
                    models[f"model_{user}"].train()
                    for images, labels in train_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = models[f"model_{user}"](images)
                        loss = criterion(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(models[f"model_{user}"].parameters(), max_norm=1.0)
                        optimizer.step()

            # Aggregate
            if participating_users:
                if global_weights is None:
                    global_weights = {}
                    for name, param in models[f"model_{participating_users[0]}"].state_dict().items():
                        global_weights[name] = torch.zeros_like(param.data)

                total_data = 0
                for user in participating_users:
                    for name, param in models[f"model_{user}"].state_dict().items():
                        global_weights[name] += param.data * datanumber[user]
                    total_data += datanumber[user]

                # Average
                for name in global_weights:
                    global_weights[name] /= total_data

                # Update global model
                global_model = NeuralNet().to(device)
                global_model.load_state_dict(global_weights)

                # Evaluate global model
                test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
                criterion = nn.CrossEntropyLoss()
                _, test_acc = evaluate(global_model, test_loader, writer, criterion, it, strategy)
                iteration_accuracy = test_acc  # keep track of last iteration accuracy

                # Sync local models
                for user in range(user_number):
                    models[f"model_{user}"].load_state_dict(global_model.state_dict())
            else:
                # If no users participated, test_acc remains as previous iteration's value
                pass

        # Store final accuracy for this strategy
        final_accuracies[strategy] = iteration_accuracy

    return final_accuracies


def main(args):
    writer = SummaryWriter()

    user_numbers = [3, 6, 9, 12, 15, 18]
    strategies = ['proposed', 'baseline1', 'baseline2', 'baseline3']

    results = {strategy: [] for strategy in strategies}

    for user_num in tqdm(user_numbers, desc="User Count Loop"):
        final_accs = run_fl_for_users(args, user_num, writer, strategies)
        for strategy in strategies:
            results[strategy].append(final_accs[strategy])

    plt.figure(figsize=(8, 6))
    for strategy in strategies:
        plt.plot(user_numbers, results[strategy], marker='o', label=strategy)
    plt.xlabel("Total number of users")
    plt.ylabel("Accuracy (%)")
    plt.title("Identification Accuracy vs Total Number of Users (R = 12)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = plt.imread(buf)
    writer.add_image('Accuracy_vs_UserCount', image, global_step=0, dataformats='HWC')
    buf.close()

    # plt.show()
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
        "--user_blocks",
        type=int,
        default=12,
        choices=[3, 6, 9, 12],
        help="number of user blocks to consider in the scenario",
    )
    args = parser.parse_args()
    main(args)

