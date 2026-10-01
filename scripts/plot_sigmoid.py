import os
import matplotlib.pyplot as plt
import numpy as np

def generate_sigmoid_plot():
    # 1. Define mathematical space
    x = np.linspace(-10, 10, 500)
    sigmoid = 1 / (1 + np.exp(-x))
    derivative = sigmoid * (1 - sigmoid)

    # 2. Configure aesthetic plotting style
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    ax.plot(x, sigmoid, label=r"$\sigma(x) = \frac{1}{1 + e^{-x}}$", color="#1f77b4", linewidth=2.5)
    ax.plot(x, derivative, label=r"$\sigma'(x) = \sigma(x)(1 - \sigma(x))$", color="#ff7f0e", linestyle="--", linewidth=2)

    # Annotations and reference lines
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.7)
    ax.axhline(1, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.7)

    ax.set_title("Sigmoid Function and Its Derivative", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Input ($x$)", fontsize=11)
    ax.set_ylabel("Output", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)

    plt.tight_layout()

    # 3. Save to assets folder relative to repository root
    output_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sigmoid_activation.png")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Generated plot saved to: {output_path}")

if __name__ == "__main__":
    generate_sigmoid_plot()
