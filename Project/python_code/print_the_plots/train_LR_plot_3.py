import argparse
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import random

from munkres import Munkres
from torch.utils.tensorboard import SummaryWriter

def main():
    writer = SummaryWriter()

    user_blocks_list = [9, 12]
    user_numbers = [5, 10, 15, 20, 25]
    results_times_ns = {9: [], 12: []}

    mnkr = Munkres()

    for ub in user_blocks_list:
        # Construct the channel_interference array
        if ub == 9:
            channel_interference = (np.array([0.03, 0.06, 0.07, 0.08, 0.1,
                                              0.11, 0.12, 0.14, 0.15]) - 0.04) * 1e-6
        else:
            channel_interference = (np.array([0.03, 0.05, 0.06, 0.07, 0.08, 0.09,
                                              0.10, 0.11, 0.12, 0.13, 0.14, 0.15]) - 0.04) * 1e-6

        num_resource_blocks = len(channel_interference)
        for user_num in user_numbers:
            # Build random cost matrix for demonstration
            W = np.random.rand(user_num, num_resource_blocks).tolist()
            
            start_ns = time.time_ns()
            _ = mnkr.compute(W)
            end_ns = time.time_ns()
            elapsed_ns = end_ns - start_ns
            results_times_ns[ub].append(elapsed_ns)

            writer.add_scalar(
                f"Munkres_Run_Time_ns/user_blocks_{ub}",
                elapsed_ns,
                user_num
            )

            print(f"[user_blocks={ub}] #Users={user_num}, time_ns={elapsed_ns}")

    plt.figure(figsize=(8, 6))
    for ub in user_blocks_list:
        plt.plot(
            user_numbers,
            results_times_ns[ub],
            marker='o',
            label=f"user_blocks = {ub}"
        )
    plt.xlabel("Number of Users")
    plt.ylabel("Time to run Munkres (ns)")
    plt.title("Hungarian (Munkres) Runtime vs Number of Users")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = plt.imread(buf)
    writer.add_image(
        "MunkresRuntime/combined_9_and_12",
        image,
        global_step=0,
        dataformats='HWC'
    )
    buf.close()

    # plt.show()

    writer.close()

if __name__ == "__main__":
    main()

