# ML-Techniques-In-Finance

This repository is dedicated to applying machine learning techniques to complex financial models. 

---
## Heston Model
This project was originally a University group project in collaboration with SS&C, "Optimisation of Stochastic Models in Finance". This repository contains some of my contributions to the project and future extensions.
The aim of this project was to calibrate Heston function parameters using machine learning. Our project ended with two methods:
1. Use a neural network to approximate the Option price of an asset with inputs: Forward Price, Strike Price, Time to Maturity, κ, θ, η, ρ, v0. Then use a recursive root finding method such as differential evolution to calibrate the Heston parameters.
2. Use a neural network to directly aproximate Heston parameters from an implied volatility surface with a list of inputs: Forward Price, Strike Price, Time to Maturity, Implied Volatility.
By utilising machine learning, we hypothesised

### 1. Option Price Approximation
A classical method 
