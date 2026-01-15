# ML-Techniques-In-Finance

This repository is dedicated to applying machine learning techniques to complex financial models. 

---
## Heston Model
This project was originally a Loughborough University group project in collaboration with SS&C, "Optimisation of Stochastic Models in Finance". This repository contains some of my contributions to the project and future extensions.
The aim of this project was to calibrate Heston function parameters using machine learning. The Heston model is stochastic volatility model, commonly applied by quantitative analysts to produce an accurate long-term implied volatility surface, to make informed decisions on an assets evaluation and respective option contracts. While the Black Scholes pricing model is constrained to a conastant volatility, the Heston model overcome this constraint with a stochastic volaitlity which more realistically found in the financial market. Due to the stochastic nature of this model, for improved accuracy, a draw back is the difficulty in computing it, requiring large integrals that for single calculations are not a problem alone but for callibrating its parameters (κ, θ, η, ρ, v0) requires monte carlo simulations or differential evolution, compounding this computational bottleneck.
Our project ended with two solutions:
1. Use a neural network to approximate the Option price of an asset with inputs: Forward Price, Strike Price, Time to Maturity, κ, θ, η, ρ, v0. Then use a recursive root finding method such as differential evolution to calibrate Heston parameters.
2. Use a neural network to directly aproximate Heston parameters from an implied volatility surface with a list of inputs: Forward Price, Strike Price, Time to Maturity, Implied Volatility.
By utilising machine learning, we hypothesised that we could greatly increase computational efficiency of callibrating these parameters than current recursive clasical methods.

### 1. Option Price Approximation
Method 1 aims to improve the speed of pricing options than the current closed form solution (only available for European options). Replacing this step in the processs with a neural network (NN) approximation can significantly increase the calculation speed, especially if we account for GPU utilisation. This method eliminates the core issue of complex mathematics and reduces the function to large numbers to linear equations, splitting the function into many simpler equations. NNs were trained using generateed synthetic data (heston_model/scripts/data_generation/random_parameters). All values are completely randomised input parameters (within realistic constraints), option price is then calculated from these variables, originally with a semi-slosed form Heston Pricing function (researched and developed by a fellow team member), later changed to Quantlib (python library) Heston pricing function for more efficient calculations in C++. MLPs and KANs were then trained with said parameters and output compaired with calculated option price to fit it's function.



### 2. Direct Heston Parameter Calibration
Method 2 poses a much larger challenge but also larger improvement. While the classical model callibration requires reccursive simulation and refineement, we investigated methods to develop generalised, trained models to callibrate directly from a whole volatility surface without slow refinement of parameters.  
