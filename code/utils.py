import numpy as np
import matplotlib.pyplot as plt
import argparse

def relational_entropy(r):
    r = np.clip(r, 1e-9, 1.0)
    return - (r * np.log(r) + (1-r) * np.log(max(1e-9, 1-r)))  # simple illustrative proxy

def make_entropy_chart(out='entropy_chart.pdf'):
    r = np.linspace(0.01, 0.99, 200)
    # Use Shannon-like single-parameter proxy for visualization
    S = -r*np.log(r)
    plt.figure(figsize=(6,4))
    plt.plot(r, S)
    plt.xlabel('Resonance (r)')
    plt.ylabel('Relational Entropy S^R (proxy)')
    plt.title('Relational Entropy decreases as resonance increases')
    plt.tight_layout()
    plt.savefig(out)
    print(f'Saved {out}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--entropy-curve', action='store_true')
    args = parser.parse_args()
    if args.entropy_curve:
        make_entropy_chart()
