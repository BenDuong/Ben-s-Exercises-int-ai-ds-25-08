import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Streamlit page setup
st.set_page_config(page_title="Law of Large Numbers: Kangaroo Ages", layout="wide")
st.title("🐾 Law of Large Numbers: Kangaroo Age Simulation")
st.markdown("Explore how sample size affects the accuracy of estimating the average age of kangaroos in Australia.")

# Sidebar controls
max_sample_size = st.sidebar.slider("Maximum Sample Size", min_value=100, max_value=2000, value=1000, step=100)
seed = st.sidebar.number_input("Random Seed", min_value=0, value=42)

# 1. Create Australian Kangaroo population
np.random.seed(seed)
population = np.random.randint(1, 20, size=10000)  # ages between 1 and 20
population_mean = np.mean(population)

st.sidebar.markdown(f"**True Population Mean Age:** `{population_mean:.2f}`")

# 2. Law of Large Numbers demonstration function
def demonstrate_lln(population, max_sample_size):
    sample_sizes = np.arange(10, max_sample_size + 1, 10)
    sample_means = [np.mean(np.random.choice(population, size=n, replace=True)) for n in sample_sizes]
    return sample_sizes, sample_means

# 3. Run simulation
sample_sizes, sample_means = demonstrate_lln(population, max_sample_size)

# 4. Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Full convergence plot
axes[0].plot(sample_sizes, sample_means, 'b-', linewidth=2, label='Sample Means')
axes[0].axhline(y=population_mean, color='red', linestyle='--', linewidth=2,
                label=f'True Population Mean ({population_mean:.2f})')
axes[0].set_title("Convergence of Sample Mean")
axes[0].set_xlabel("Sample Size")
axes[0].set_ylabel("Sample Mean Age")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Zoomed-in view
start_idx = len(sample_sizes) // 2
axes[1].plot(sample_sizes[start_idx:], sample_means[start_idx:], 'b-', linewidth=2, label='Sample Means')
axes[1].axhline(y=population_mean, color='red', linestyle='--', linewidth=2, label='Population Mean')
axes[1].set_title("Zoomed: Later Convergence")
axes[1].set_xlabel("Sample Size")
axes[1].set_ylabel("Sample Mean Age")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

st.pyplot(fig)

# 5. Numerical convergence table
st.subheader("📐 Convergence Analysis")
checkpoints = [50, 100, 200, 500, 1000]
data = []

for n in checkpoints:
    if n <= max(sample_sizes):
        idx = np.where(sample_sizes == n)[0][0]
        sample_mean = sample_means[idx]
        distance = abs(sample_mean - population_mean)
        data.append((n, round(sample_mean, 2), round(distance, 3)))

st.table(
    {
        "Sample Size": [row[0] for row in data],
        "Sample Mean": [row[1] for row in data],
        "Distance from True Mean": [row[2] for row in data]
    }
)

st.success("🎯 As sample size increases, the sample mean converges toward the true population mean.")