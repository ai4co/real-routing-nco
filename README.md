# Real Routing NCO

[![arXiv](https://img.shields.io/badge/arXiv-2503.16159-b31b1b.svg)](https://arxiv.org/abs/2503.16159)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

> Note: we are currently finalizing the repository. Stay tuned!


### 🗺️ Problem
Most NCO methods use simplified routing with 2D Euclidean distance. This is not realistic for real-world applications which can have complex 1) _distance matrices_ and 2) _duration matrices_ between locations because of road networks, traffic, and more.

<p align="center">
  <img src="assets/simple_routing.png" width="300" /><img src="assets/real_routing.png" width="300" />
<br>
  <em>Left: previous works with simplified routing. Right: RRNCO with real-world routing!</em>
  <br>
</p>

How can we bridge this gap between toy and real settings?

We need two things:
1) A **dataset** with real-world routing information
2) A **model** that can handle such data -- not only node but also _edge_ information




### ✨ Solution 1: RRNCO Dataset

We introduce the RRNCO (Real Routing NCO) dataset, which contains real-world routing information for 100 cities around the world, from which instances can be subsampled and generated on the fly

<p align="center">
  <img src="assets/data_generation.png" />
  <br>
    <em>RRNCO data generation pipeline</em>
<br>

### ✨ Solution 2: The RRNCO Model

The RRNCO model efficiently processes topology information by leveraging several techniques including scale adaptive biases

<p align="center">
  <img src="assets/model.png" />
  <br>
    <em>RRNCO model architecture</em>
<br>



## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for faster installation and dependency management. To install it, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, clone the repository and cd into it:
```bash
git clone git@github.com:ai4co/real-routing-nco.git
cd real-routing-nco
```

Create a new virtual environment and activate it:
```bash
uv venv --python 3.12
source .venv/bin/activate
```

Then synchronize the dependencies:
```bash
uv sync --all-groups
```

### Data download

TODO (HuggingFace)

### Model checkpoints

TODO (HuggingFace)


### Data generation

Instructions on how to install the OSRM backend and generate (new) datasets [data_generation](data_generation/README.md) folder.


### How to run

To get started with running RRNCO, please follow the steps below:

---
**1. Prepare the dataset**

After generating city data using the data generation pipeline, move the generated files to the following directory:

`data/dataset/{city}/{city}_data.npz`

For example, if the city is Seoul, the data file should be located at:

`data/dataset/Seoul/Seoul_data.npz`

Additionally, the file `data/dataset/splited_cities_list.json` contains a predefined split of cities into training and test sets. If you wish to modify the training cities, simply edit the list under the `"train"` key in this JSON file.

**2. Generate validation dataset for training**

To generate validation data (used during training), run:

```bash
python generate_data.py
```

**3. Generate test dataset**

To generate the test dataset (used during evaluation with `test.py`), run:


```bash
python generate_data.py --seed 3333

```

**4. Generate test dataset**

To train a model, use the `train.py` script. For example, to train a model for the ATSP problem:

```bash
python train.py experiment=rrnet env=atsp
```
Available environment options are:

- atsp (Asymmetric TSP)

- rcvrp (Real-world Capacitated VRP(ACVRP))

- rcvrptw (Real-world Capacitated VRP with Time Windows(ACVRPTW))

You can also configure experiment settings using the file `config/experiment/rrnet.yaml`.

**5. Evaluate the model**

You can evaluate a trained model using the `test.py` script. Make sure to provide the correct dataset path via `--datasets` and model checkpoint via `--checkpoint`.

Examples for different tasks:

**ATSP**
```bash
python test.py --problem atsp --datasets data/atsp/atsp_n100_seed3333_in_distribution.npz --batch_size 32 --checkpoint checkpoints/atsp/epoch_199.ckpt
```

**RCVRP**
```bash
python test.py --problem rcvrp --datasets data/rcvrp/rcvrp_n100_seed3333_in_distribution.npz --batch_size 32 --checkpoint checkpoints/rcvrp/epoch_199.ckpt
```

**RCVRPTW**
```bash
python test.py --problem rcvrptw --datasets data/rcvrptw/rcvrptw_n100_seed3333_in_distribution.npz --batch_size 32 --checkpoint checkpoints/rcvrptw/epoch_199.ckpt
```



### 🤩 Citation
If you find RRNCO valuable for your research or applied projects:

```bibtex
@article{son2025rrnco_neuralcombinatorialoptimizationrealworldrouting,
      title={{Neural Combinatorial Optimization for Real-World Routing}},
      author={Jiwoo Son and Zhikai Zhao and Federico Berto and Chuanbo Hua and Changhyun Kwon and Jinkyoo Park},
      year={2025},
      eprint={2503.16159},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://github.com/ai4co/real-routing-nco},
}
```
