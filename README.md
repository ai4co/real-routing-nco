# Real Routing NCO

### ❗ Problem
Most NCO methods use simplified routing with 2D Euclidean distance. This is not realistic for real-world applications which can have complex 1) _distance matrices_ and 2) _duration matrices_ between locations because of road networks, traffic, and more.

<p align="center">
  <img src="assets/simple_routing.png" width="150" /><img src="assets/real_routing.png" width="150" />
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

TODO


### Citation

TODO


---

---

---
