# ML-Techniques-In-Finance

This repository is dedicated to applying machine learning techniques to complex financial models. 

---
## Heston Model
This project was originally a University group project in collaboration with SS&C, "Optimisation of Stochastic Models in Finance". This repository contains some of my contributions to the project and future extensions.
The aim of this project was to calibrate Heston function parameters using machine learning. The Heston model is stochastic volatility model, commonly applied by quantitative analysts to produce an accurate long-term implied volatility surface, to make informed decisions on an assets evaluation and respective option contracts. The model is an extension of the Black Acholes pricing model that is constrained.Due to the stochastic nature of this model, for improved accuracy, a draw back is the difficulty in computing it, requiring large integrals that fro single calculations are not a problem but for callibrating its parameter (κ, θ, η, ρ, v0) require monte carlo simulations or
Our project ended with two methods:
1. Use a neural network to approximate the Option price of an asset with inputs: Forward Price, Strike Price, Time to Maturity, κ, θ, η, ρ, v0. Then use a recursive root finding method such as differential evolution to calibrate the Heston parameters.
2. Use a neural network to directly aproximate Heston parameters from an implied volatility surface with a list of inputs: Forward Price, Strike Price, Time to Maturity, Implied Volatility.
By utilising machine learning, we hypothesised that we could greatly increase computational efficiency of callibrating these parameters than current recursive clasical methods.

### 1. Option Price Approximation
Method 1 aims to improve the speed of pricing options than the current closed form solution for European options, which is reliant on complex integrations. 
