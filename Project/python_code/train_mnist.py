import argparse
import random
from collections import defaultdict
import matplotlib.pyplot as plt

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

def simple_logger(message):
    print(message)

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
    def __init__(self, input_size=28*28, hidden_size=50, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def packet_error_calculator(distance_matrix, channel_interference, user_max_power):
    epsilon = 1e-6
    distance_matrix = np.maximum(distance_matrix, epsilon)
    q_packet_error = 1 - np.exp(-1.08 * (channel_interference + 1e-14) / (user_max_power * np.power(distance_matrix, -2)))
    return q_packet_error


def sinr_calculator(distance_matrix, channel_interference_matrix, power=1):
    epsilon = 1e-6
    distance_matrix = np.maximum(distance_matrix, epsilon)
    SINR = power * np.divide(np.power(distance_matrix, -2), channel_interference_matrix + epsilon)
    return SINR


def per_user_total_energy_calculator(fl_model_data_size, uplink_delay, psi, omega_i, theta, user_power):
    total_energy = psi * omega_i * (theta ** 2) * fl_model_data_size + user_power * uplink_delay
    return total_energy


def channel_rate_calculator(bandwidth, sinr):
    sinr = np.clip(sinr, 1e-6, 1e6)
    rate = bandwidth * np.log2(1 + sinr)
    return rate


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def evaluate(model, test_loader, writer, criterion, iteration, strategy):
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

    simple_logger(f'[{strategy}] Iteration {iteration}: Test Accuracy: {test_accuracy:.2f}%, Loss: {average_loss:.4f}')
    writer.add_scalar(f"{strategy}/accuracy/test_global_model", test_accuracy, global_step=iteration)
    writer.add_scalar(f"{strategy}/loss/test_global_model", average_loss, global_step=iteration)
    print(f"[{strategy}] Iteration {iteration}: Accuracy={test_accuracy:.2f}%, Loss={average_loss:.4f}")

    model.train()
    return average_loss


def train(args):
    mnkr = Munkres()
    # Number of training datasamples for each device.
    datanumber = [100, 200, 300, 400, 500, 400, 300, 200, 100, 200, 300, 400, 500, 600, 100, 200, 300, 400, 500, 100]

    channel_interference_downlink = 0.06 * 0.000003

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

    if args.user_blocks == 3:
        channel_interference = (np.array([0.05, 0.1, 0.14]) - 0.04) * 0.000001
    elif args.user_blocks == 6:
        channel_interference = (np.array([0.05, 0.07, 0.09, 0.11, 0.13, 0.15]) - 0.04) * 0.000001
    elif args.user_blocks == 9:
        channel_interference = (np.array([0.03, 0.06, 0.07, 0.08, 0.1, 0.11, 0.12, 0.14, 0.15]) - 0.04) * 0.000001
    elif args.user_blocks == 12:
        channel_interference = (np.array([0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]) - 0.04) * 0.000001
    else:
        raise ValueError("Unsupported number of user blocks.")

    num_resource_blocks = len(channel_interference)
    writer = SummaryWriter()

    strategies = ['proposed', 'baseline1', 'baseline2', 'baseline3']
    error_dict = {strategy: [] for strategy in strategies}

    for average in range(args.averagenumber):
        distance = np.random.rand(args.user_number, 1) * 500
        q_packet_error = packet_error_calculator(distance, channel_interference, config['channel_parameters']['user_max_power'])
        sinr = sinr_calculator(distance, channel_interference, config["channel_parameters"]["user_max_power"])
        rateu = channel_rate_calculator(
            config['channel_parameters']["uplink_bandwidth"],
            sinr
        )

        SINRd = sinr_calculator(distance, channel_interference_downlink)
        rated = channel_rate_calculator(
            config['channel_parameters']['downlink_bandwidth'],
            SINRd
        )
        
        models = {strategy: {} for strategy in strategies}
        for strategy in strategies:
            for user in range(args.user_number):
                models[strategy][f"model_{user}"] = NeuralNet(input_size=28*28, hidden_size=50, num_classes=10).to(device)
                models[strategy][f"model_{user}"].apply(init_weights)

        total_model_params = sum(p.numel() for p in models[strategies[0]]['model_0'].parameters())
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


        global_weights = {strategy: None for strategy in strategies}

        error_tracking = {strategy: np.zeros((args.iteration, 1)) for strategy in strategies}

        for strategy in strategies:
            global_weights[strategy] = None
            for user in range(args.user_number):
                models[strategy][f"model_{user}"].train()

        # Start FL iterations for each strategy
        for strategy in strategies:
            simple_logger(f"Starting strategy: {strategy}")
            W = np.zeros((args.user_number, num_resource_blocks))
            finalq = np.ones((1, args.user_number))

            # Resource Allocation per strategy (recompute for each average and strategy)
            if strategy == 'proposed':
                for i in range(args.user_number):
                    for j in range(num_resource_blocks):
                        if (totaldelay[i, j] < config['channel_parameters']['delay_requirement']) and \
                           (totalenergy[i, j] < config['channel_parameters']['energy_requirement']):
                            W[i, j] = -datanumber[i] * (1 - q_packet_error[i, j])
                        else:
                            W[i, j] = 1e+10

                # Use Hungarian algorithm to find the optimal RB allocation
                try:
                    assignment = mnkr.compute(W.tolist())
                except Exception as e:
                    simple_logger(f"[{strategy}] Hungarian Algorithm failed: {e}")
                    assignment = []

                for assign in assignment:
                    if len(assign) == 2:
                        i, j = assign
                        if W[i][j] != 1e+10:
                            finalq[0, i] = q_packet_error[i, j]
                        else:
                            finalq[0, i] = 1

            elif strategy == 'baseline1':
                for i in range(args.user_number):
                    for j in range(num_resource_blocks):
                        if (totaldelay[i, j] < config['channel_parameters']['delay_requirement']) and \
                           (totalenergy[i, j] < config['channel_parameters']['energy_requirement']):
                            W[i, j] = -datanumber[i] * (1 - q_packet_error[i, j])
                        else:
                            W[i, j] = 1e+10

                # Use Hungarian algorithm to find the optimal RB allocation
                try:
                    assignment = mnkr.compute(W.tolist())
                except Exception as e:
                    simple_logger(f"[{strategy}] Hungarian Algorithm failed: {e}")
                    assignment = []

                # Perform random RB allocation for the assigned users
                qassignment = np.zeros((1, args.user_number))
                assigned_users = []
                assigned_rbs = []
                for assign in assignment:
                    if len(assign) == 2:
                        i, j = assign
                        if W[i][j] != 1e+10:
                            assigned_users.append(i)
                            assigned_rbs.append(j)

                if len(assigned_users) > 0:
                    selected_rbs = random.sample(range(num_resource_blocks), len(assigned_users))
                    for idx, user in enumerate(assigned_users):
                        qassignment[0, user] = selected_rbs[idx]

                # Calculate finalq based on qassignment
                for user in range(args.user_number):
                    j = int(qassignment[0, user])
                    if j < num_resource_blocks and W[user, j] != 1e+10:
                        finalq[0, user] = q_packet_error[user, j]
                    else:
                        finalq[0, user] = 1  # Not assigned

            elif strategy == 'baseline2':
                # Random RB allocation and user selection
                qassignment = np.zeros((1, args.user_number))
                assignment = np.zeros((1, args.user_number))

                if num_resource_blocks < args.user_number:
                    selected_users = random.sample(range(args.user_number), num_resource_blocks)
                else:
                    selected_users = list(range(args.user_number))

                selected_rbs = random.sample(range(num_resource_blocks), len(selected_users))

                for idx, user in enumerate(selected_users):
                    assignment[0, user] = 1
                    qassignment[0, user] = selected_rbs[idx]

                # Calculate finalq based on qassignment
                for user in range(args.user_number):
                    j = int(qassignment[0, user])
                    if j < num_resource_blocks and assignment[0, user] == 1:
                        if (totaldelay[user, j] < config['channel_parameters']['delay_requirement']) and \
                           (totalenergy[user, j] < config['channel_parameters']['energy_requirement']):
                            finalq[0, user] = q_packet_error[user, j]
                        else:
                            finalq[0, user] = 1  # Constraints not met
                    else:
                        finalq[0, user] = 1  # Not assigned

            elif strategy == 'baseline3':
                for i in range(args.user_number):
                    for j in range(num_resource_blocks):
                        if (totaldelay[i, j] < config['channel_parameters']['delay_requirement']) and \
                           (totalenergy[i, j] < config['channel_parameters']['energy_requirement']):
                            W[i, j] = q_packet_error[i, j]
                        else:
                            W[i, j] = 1e+10

                # Use Hungarian algorithm to find the optimal RB allocation
                try:
                    assignment = mnkr.compute(W.tolist())
                except Exception as e:
                    simple_logger(f"[{strategy}] Hungarian Algorithm failed: {e}")
                    assignment = []

                # Perform random RB allocation for the assigned users
                qassignment = np.zeros((1, args.user_number))
                assigned_users = []
                assigned_rbs = []
                for assign in assignment:
                    if len(assign) == 2:
                        i, j = assign
                        if W[i][j] != 1e+10:
                            assigned_users.append(i)
                            assigned_rbs.append(j)

                if len(assigned_users) > 0:
                    if num_resource_blocks < len(assigned_users):
                        selected_rbs = random.sample(range(num_resource_blocks), len(assigned_users))
                    else:
                        selected_rbs = random.sample(range(num_resource_blocks), len(assigned_users))
                    for idx, user in enumerate(assigned_users):
                        qassignment[0, user] = selected_rbs[idx]

                # Calculate finalq based on qassignment
                for user in range(args.user_number):
                    j = int(qassignment[0, user])
                    if j < num_resource_blocks and W[user, j] != 1e+10:
                        finalq[0, user] = q_packet_error[user, j]
                    else:
                        finalq[0, user] = 1  # Not assigned

            # Initialize participation indicator
            participation = np.zeros((args.iteration, args.user_number))
            error = np.zeros((args.iteration, 1))

            # Initialize global weights
            global_weights[strategy] = None

            # Initialize models for this strategy
            strategy_models = models[strategy]

            # Initialize global_weights as zero tensors
            for user in range(args.user_number):
                strategy_models[f"model_{user}"].apply(init_weights)

            for iteration in tqdm(range(args.iteration), desc=f"Strategy: {strategy}"):
                participating_users = []

                for user in range(args.user_number):
                    if (iteration == 0 and finalq[0, user] != 1) or (random.random() > finalq[0, user]):
                        participating_users.append(user)
                        participation[iteration, user] = 1

                        # Set input data
                        start_idx = sum(datanumber[:user])
                        end_idx = sum(datanumber[:user + 1])
                        user_train_dataset = torch.utils.data.Subset(train_dataset, list(range(start_idx, end_idx)))
                        train_loader = DataLoader(dataset=user_train_dataset, batch_size=args.batch_size, shuffle=True)
                        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(strategy_models[f"model_{user}"].parameters(), lr=0.001)  # Reduced learning rate

                        # Training loop
                        strategy_models[f"model_{user}"].train()
                        for images, labels in train_loader:
                            images = images.to(device)
                            labels = labels.to(device)

                            # Forward pass
                            outputs = strategy_models[f"model_{user}"](images)
                            loss = criterion(outputs, labels)

                            # Backward and optimize
                            optimizer.zero_grad()
                            loss.backward()

                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(strategy_models[f"model_{user}"].parameters(), max_norm=1.0)

                            optimizer.step()

                # Aggregate global model
                if participating_users:
                    # Initialize global weights as zeros
                    if global_weights[strategy] is None:
                        global_weights[strategy] = {}
                        for name, param in strategy_models[f"model_{participating_users[0]}"].state_dict().items():
                            global_weights[strategy][name] = torch.zeros_like(param.data)

                    # Sum the weights from participating users
                    for user in participating_users:
                        for name, param in strategy_models[f"model_{user}"].state_dict().items():
                            global_weights[strategy][name] += param.data * datanumber[user]

                    # Average the weights
                    total_datanumber = sum([datanumber[user] for user in participating_users])
                    for name in global_weights[strategy]:
                        global_weights[strategy][name] /= total_datanumber

                    # Update global model
                    global_model = NeuralNet(input_size=28*28, hidden_size=50, num_classes=10).to(device)
                    global_model.load_state_dict(global_weights[strategy])

                    # Evaluate the global model
                    average_loss = evaluate(global_model, test_loader, writer, criterion, iteration, strategy)

                    # Update local models to the global model
                    for user in range(args.user_number):
                        strategy_models[f"model_{user}"].load_state_dict(global_model.state_dict())

                    # Record the error
                    error[iteration] = average_loss
                else:
                    # If no users participated, carry forward the previous error
                    if iteration > 0:
                        error[iteration] = error[iteration - 1]
                    else:
                        error[iteration] = 1.0

                # simple_logger(f"[{strategy}] Iteration {iteration}: Participating Users: {participating_users}")

            # Record the final error for this strategy and average
            error_dict[strategy].append(error[-1, 0])

    # After all averages, compute the mean error for each strategy
    finalerror = {}
    for strategy in strategies:
        finalerror[strategy] = np.mean(error_dict[strategy])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iteration",
        type=int,
        default=130,
        help="number of total iterations",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size of the model",
    )

    parser.add_argument(
        "--averagenumber",
        type=int,
        default=3,
        help="Number of implementations of FL",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.003,
        help="learning rate",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default='proposed',
        help="the strategy of FL training",
        choices=['proposed', 'baseline1', 'baseline2', 'baseline3']
    )

    parser.add_argument(
        "--user_number",
        type=int,
        default=12,
        help="Total number of users that implement FL",
    )

    parser.add_argument(
        "--user_blocks",
        type=int,
        default=6,
        choices=[3, 6, 9, 12],
        help="Total number of user blocks that implement FL",
    )

    args = parser.parse_args()
    train(args)
