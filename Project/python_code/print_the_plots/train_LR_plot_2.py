import argparse
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import io

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import torch.optim as optim
from munkres import Munkres
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


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


def generate_data(num_samples=50):
    """
    Generates synthetic linear regression data.
    y = -2x + 1 + noise * 0.4
    Clamped to y ∈ [-2, 2].
    """
    x = np.random.rand(num_samples, 1).astype(np.float32)  # x ∈ [0,1]
    noise = np.random.normal(0, 1, size=(num_samples, 1)).astype(np.float32)
    y = -2 * x + 1 + noise * 0.4
    y = np.clip(y, -2, 2)  # y ∈ [-2,2]
    return torch.tensor(x), torch.tensor(y)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        return self.fc2(x)

def plot_loss_vs_samples(writer, sample_sizes, error_dict):
    """
    Plots the global loss against the number of data samples per user for all strategies.
    Logs the plot to TensorBoard.
    """
    plt.figure(figsize=(10, 8))
    
    colors = {
        'proposed': 'blue',
        'baseline1': 'green',
        'baseline2': 'orange',
        'baseline3': 'purple'
    }
    
    for strategy, losses in error_dict.items():
        plt.plot(sample_sizes, losses, marker='o', color=colors[strategy], label=strategy)
    
    plt.xlabel('Number of Data Samples per User')
    plt.ylabel('Value of the Global Loss Function')
    plt.title('Global Loss vs Number of Data Samples per User')
    plt.legend()
    plt.grid(True)
    
    # Convert plot to TensorBoard image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    writer.add_image('Global Loss vs Data Samples', np.array(image), global_step=0, dataformats='HWC')
    plt.close()


def packet_error_calculator(distance_matrix, channel_interference, user_max_power):
    """
    Calculates packet error rates based on distance and interference.
    """
    epsilon = 1e-6
    distance_matrix = np.maximum(distance_matrix, epsilon)
    q_packet_error = 1 - np.exp(-1.08 * (channel_interference + 1e-14) / (user_max_power * np.power(distance_matrix, -2)))
    return q_packet_error

def sinr_calculator(distance_matrix, channel_interference_matrix, power=1):
    """
    Calculates SINR (Signal-to-Interference-plus-Noise Ratio).
    """
    epsilon = 1e-6
    distance_matrix = np.maximum(distance_matrix, epsilon)
    SINR = power * np.divide(np.power(distance_matrix, -2), channel_interference_matrix + epsilon)
    return SINR

def per_user_total_energy_calculator(fl_model_data_size, uplink_delay, psi, omega_i, theta, user_power):
    """
    Calculates total energy consumption per user.
    """
    total_energy = psi * omega_i * (theta ** 2) * fl_model_data_size + user_power * uplink_delay
    return total_energy

def channel_rate_calculator(bandwidth, sinr):
    """
    Calculates channel rate based on bandwidth and SINR.
    """
    sinr = np.clip(sinr, 1e-6, 1e6)
    rate = bandwidth * np.log2(1 + sinr)
    return rate

def init_weights(m):
    """
    Initializes model weights using Xavier uniform initialization.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def evaluate(model, test_loader, writer, criterion, iteration, strategy):
    """
    Evaluates the model on the test dataset and logs the results.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

    average_loss = total_loss / len(test_loader.dataset)

    simple_logger(f'[{strategy}] Iteration {iteration}: Test MSE Loss: {average_loss:.4f}')
    writer.add_scalar(f"{strategy}/loss/test_global_model", average_loss, global_step=iteration)
    print(f"[{strategy}] Iteration {iteration}: MSE Loss={average_loss:.4f}")

    model.train()
    return average_loss


def train(args):
    mnkr = Munkres()
    # Interference over downlink
    channel_interference_downlink = 0.06 * 0.000003

    # Define channel interference based on user blocks
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

    sample_sizes = [10, 20, 30, 40, 50]

    # Iterate over different sample sizes
    for size in sample_sizes:
        simple_logger(f"\n=== Starting FL Training with {size} Samples per User ===\n")
        datanumber = [size] * args.user_number

        distance = np.random.rand(args.user_number, 1) * 500
        q_packet_error = packet_error_calculator(distance, channel_interference, config['channel_parameters']['user_max_power'])
        sinr = sinr_calculator(distance, channel_interference, config["channel_parameters"]["user_max_power"])
        rateu = channel_rate_calculator(config['channel_parameters']["uplink_bandwidth"], sinr)

        SINRd = sinr_calculator(distance, channel_interference_downlink, config['channel_parameters']['user_max_power'])
        rated = channel_rate_calculator(config['channel_parameters']['downlink_bandwidth'], SINRd)

        # Initialize neural networks for each strategy and user
        models = {strategy: {} for strategy in strategies}
        for strategy in strategies:
            for user in range(args.user_number):
                models[strategy][f"model_{user}"] = LinearRegressionModel().to(device)
                models[strategy][f"model_{user}"].apply(init_weights)

        total_model_params = sum(p.numel() for p in models[strategies[0]]['model_0'].parameters())
        Z = total_model_params * 16 / 1024 / 1024

        delayu = Z / rateu
        delayd = Z / rated
        totaldelay = delayu + delayd
        totalenergy = per_user_total_energy_calculator(Z, delayu, config['channel_parameters']['psi'], config['channel_parameters']['omega_i'], config['channel_parameters']['theta'], config["channel_parameters"]["user_max_power"])

        global_weights = {strategy: None for strategy in strategies}

        for strategy in strategies:
            W = np.zeros((args.user_number, num_resource_blocks))
            finalq = np.ones((1, args.user_number))

            if strategy == 'proposed':
                for i in range(args.user_number):
                    for j in range(num_resource_blocks):
                        if (totaldelay[i, j] < config['channel_parameters']['delay_requirement']) and \
                           (totalenergy[i, j] < config['channel_parameters']['energy_requirement']):
                            W[i, j] = -datanumber[i] * (1 - q_packet_error[i, j])
                        else:
                            W[i, j] = 1e+10

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

                try:
                    assignment = mnkr.compute(W.tolist())
                except Exception as e:
                    simple_logger(f"[{strategy}] Hungarian Algorithm failed: {e}")
                    assignment = []

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

                for user in range(args.user_number):
                    j = int(qassignment[0, user])
                    if j < num_resource_blocks and W[user, j] != 1e+10:
                        finalq[0, user] = q_packet_error[user, j]
                    else:
                        finalq[0, user] = 1

            elif strategy == 'baseline2':
                qassignment = np.zeros((1, args.user_number))
                assignment_matrix = np.zeros((1, args.user_number))

                if num_resource_blocks < args.user_number:
                    selected_users = random.sample(range(args.user_number), num_resource_blocks)
                else:
                    selected_users = list(range(args.user_number))

                selected_rbs = random.sample(range(num_resource_blocks), len(selected_users))

                for idx, user in enumerate(selected_users):
                    assignment_matrix[0, user] = 1
                    qassignment[0, user] = selected_rbs[idx]

                for user in range(args.user_number):
                    j = int(qassignment[0, user])
                    if j < num_resource_blocks and assignment_matrix[0, user] == 1:
                        if (totaldelay[user, j] < config['channel_parameters']['delay_requirement']) and \
                           (totalenergy[user, j] < config['channel_parameters']['energy_requirement']):
                            finalq[0, user] = q_packet_error[user, j]
                        else:
                            finalq[0, user] = 1
                    else:
                        finalq[0, user] = 1

            elif strategy == 'baseline3':
                for i in range(args.user_number):
                    for j in range(num_resource_blocks):
                        if (totaldelay[i, j] < config['channel_parameters']['delay_requirement']) and \
                           (totalenergy[i, j] < config['channel_parameters']['energy_requirement']):
                            W[i, j] = q_packet_error[i, j]
                        else:
                            W[i, j] = 1e+10

                try:
                    assignment = mnkr.compute(W.tolist())
                except Exception as e:
                    simple_logger(f"[{strategy}] Hungarian Algorithm failed: {e}")
                    assignment = []

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

                for user in range(args.user_number):
                    j = int(qassignment[0, user])
                    if j < num_resource_blocks and W[user, j] != 1e+10:
                        finalq[0, user] = q_packet_error[user, j]
                    else:
                        finalq[0, user] = 1

            participation = np.zeros((args.iteration, args.user_number))
            error = np.zeros((args.iteration, 1))

            strategy_models = models[strategy]
            global_weights[strategy] = None

            for iteration in tqdm(range(args.iteration), desc=f"Strategy: {strategy} | Sample Size: {size}"):
                participating_users = []

                for user in range(args.user_number):
                    participation_probability = 1 - finalq[0, user]
                    if random.random() < participation_probability:
                        participating_users.append(user)
                        participation[iteration, user] = 1

                        x_train, y_train = generate_data(num_samples=datanumber[user])
                        train_dataset = TensorDataset(x_train, y_train)
                        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

                        criterion = nn.MSELoss()
                        optimizer = optim.Adam(strategy_models[f"model_{user}"].parameters(), lr=args.learning_rate)

                        strategy_models[f"model_{user}"].train()
                        for images, labels in train_loader:
                            images = images.to(device)
                            labels = labels.to(device)

                            outputs = strategy_models[f"model_{user}"](images)
                            loss = criterion(outputs, labels)

                            optimizer.zero_grad()
                            loss.backward()

                            torch.nn.utils.clip_grad_norm_(strategy_models[f"model_{user}"].parameters(), max_norm=1.0)

                            optimizer.step()

                if participating_users:
                    if global_weights[strategy] is None:
                        global_weights[strategy] = {}
                        for name, param in strategy_models[f"model_{participating_users[0]}"].state_dict().items():
                            global_weights[strategy][name] = torch.zeros_like(param.data)

                    for user in participating_users:
                        for name, param in strategy_models[f"model_{user}"].state_dict().items():
                            global_weights[strategy][name] += param.data * datanumber[user]

                    total_datanumber = sum([datanumber[user] for user in participating_users])
                    for name in global_weights[strategy]:
                        global_weights[strategy][name] /= total_datanumber

                    global_model = LinearRegressionModel().to(device)
                    global_model.load_state_dict(global_weights[strategy])

                    x_plot = np.linspace(0, 1, 100).reshape(-1,1).astype(np.float32)
                    y_test = -2 * x_plot + 1 + np.random.normal(0, 0.4, x_plot.shape).astype(np.float32)
                    y_test = np.clip(y_test, -2, 2)
                    x_test_tensor = torch.tensor(x_plot)
                    y_test_tensor = torch.tensor(y_test)
                    test_loader = DataLoader(dataset=TensorDataset(x_test_tensor, y_test_tensor), batch_size=args.batch_size, shuffle=False)

                    average_loss = evaluate(global_model, test_loader, writer, criterion, iteration, strategy)

                    for user in range(args.user_number):
                        strategy_models[f"model_{user}"].load_state_dict(global_model.state_dict())

                    error[iteration] = average_loss
                else:
                    if iteration > 0:
                        error[iteration] = error[iteration - 1]
                    else:
                        error[iteration] = 1.0

            final_loss = error[-1, 0]
            error_dict[strategy].append(final_loss)
            simple_logger(f"[{strategy}] Final Loss for sample size {size}: {final_loss:.4f}")

    plot_loss_vs_samples(writer, sample_sizes, error_dict)

    simple_logger("\nFinal MSE Losses for Strategies across Sample Sizes:")
    for strategy in strategies:
        losses = error_dict[strategy]
        loss_str = ", ".join([f"{size}: {loss:.4f}" for size, loss in zip(sample_sizes, losses)])
        simple_logger(f"{strategy}: {loss_str}")

    writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iteration",
        type=int,
        default=130,
        help="Number of total FL iterations",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default='proposed',
        help="Strategy of FL training",
        choices=['proposed', 'baseline1', 'baseline2', 'baseline3']
    )

    parser.add_argument(
        "--user_number",
        type=int,
        default=12,
        help="Total number of users participating in FL",
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

