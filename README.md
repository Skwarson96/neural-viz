# neural-viz

This project demonstrates the process of training a neural network to 
fit various mathematical functions, including sine, rectangle, sawtooth,
and polynomial functions. The project allows for the visualization of the 
learning process through animated plots.

## Function Visualizations

### Sinusoidal Function
<p align="center">
  <img src="https://github.com/Skwarson96/neural-viz/blob/main/assets/sin_animation.gif"/>
</p>

### Rectangle Function
<p align="center">
  <img src="https://github.com/Skwarson96/neural-viz/blob/main/assets/rectangle_animation.gif"/>
</p>

### Sawtooth Function
<p align="center">
  <img src="https://github.com/Skwarson96/neural-viz/blob/main/assets/sawtooth_animation.gif"/>
</p>

### Polynomial Function
<p align="center">
  <img src="https://github.com/Skwarson96/neural-viz/blob/main/assets/polynomial_animation.gif"/>
</p>

## Description

This project uses a simple neural network implemented in PyTorch to fit
different types of functions. The functions include sinusoidal, rectangle,
sawtooth, and polynomial functions. The neural network's learning process is
visualized using animated plots created with Matplotlib.

## Getting Started

### Prerequisites
You can install the required packages using pip:
```sh
pip install -r requirements.txt
```

# Usage
Run the script with the desired parameters to generate and visualize the function
fitting process. Below is the basic usage with all available arguments explained.
```
python main.py --start <START> --stop <STOP> --resolution <RESOLUTION> --amplitude <AMPLITUDE> --function <FUNCTION> --epochs <EPOCHS> --interval <INTERVAL> --noise_level <NOISE_LEVEL> --save
```

# Arguments
- --start: X axis start value.
- --stop: X axis end value.
- --resolution: Resolution of the generated data.
- --amplitude: Amplitude of the sinusoidal function.
- --function: Type of function to learn. Options: sin, rectangle, sawtooth, polynomial. Default: sin.
- --epochs: Number of training epochs.
- --interval: Interval for plotting results.
- --noise_level: Noise level added to the data.
- --save: Save gif file