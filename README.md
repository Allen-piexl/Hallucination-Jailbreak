From Hallucinations to Jailbreaks: Rethinking the Vulnerability of Large Foundation Models

## Environment Setup

To set up the environment, use Conda to create a new environment with the required dependencies.

### Step 1: Create a Conda Environment
```bash
conda create -n hallucination-exp python=3.10
conda activate hallucination-exp
pip install torch torchvision transformers openai==0.28.0
```

### Step 2: Attacker
```bash
python attacker.py
```

# Note
This is a simple example. We will improve the repository later.

See the paper for more details.
