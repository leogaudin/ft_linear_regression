<h1 align='center'>📈 ft_linear_regression</h1>

**`ft_linear_regression` is the first project of the Machine Learning branch at 42.**

The goal of this project is to introduce us to the basic concepts of ML.

We will have to implement a program that predicts the price of a car by using a linear function train with a gradient descent algorithm.

*The implementation of this project will be done in Python.*

## Table of Contents

- [Concepts needed for the project](#concepts-needed-for-the-project)
	- [Linear Regression](#linear-regression)
	- [Gradient Descent](#gradient-descent)
	- [Loss Function](#loss-function)
	- [Feature Scaling](#feature-scaling)
	- [Normal Equation](#normal-equation)
- [Resources](#resources) 📖

# Concepts needed for the project

## Linear Regression

Simple linear regression aims to represent the relationship between a feature and a target variable.

The formula for a linear regression is:

$$
y = θ_{0} + θ_{1}x
$$

Where:

- $y$ is the target variable
- $x$ is the feature
- $θ_{0}$ is the bias
- $θ_{1}$ is the coefficient

It's literally a line equation, like $y = mx + b$ in school.

In our case, the target variable is the price of a car and the feature is the mileage.

> For example, if we find out that $price = -0.5 * mileage$, it means that for each unit of mileage, the price of the car decreases by 0.5.

## Loss Function

In ML, a loss function is used to quantify how well a model is performing, its precision.

It measures the **difference between the predicted values and the actual values**.

The loss function used in our linear regression is the **Mean Absolute Error** (MAE).

$$
MAE = \frac{1}{m}  \sum_{i=1}^{m} |y_{i} - \hat{y}_{i}|
$$

Where:

- $m$ is the number of samples
- $y$ is the actual value
- $\hat{y}$ is the predicted value

To make it simpler, **it means we are going to estimate the price for a mileage, subtract it to the actual price in the data we have, do this for all the data points, and then divide the sum by the number of data points**.

For example, if we have 3 data points:

- $y_{1} = 1000$ and $\hat{y}_{1} = 900$
- $y_{2} = 2000$ and $\hat{y}_{2} = 1900$
- $y_{3} = 3000$ and $\hat{y}_{3} = 3100$

The MAE will be:

$$
MAE = \frac{|1000 - 900| + |2000 - 1900| + |3000 - 3100|}{3} = 100
$$

Here, the MAE is 100, which means that our model is off by 100 on average.

> Unless your data points are perfectly aligned, the MAE will never be 0, it is totally normal to have a loss.

Now that we know how to measure the precision of our model, we need to find the best $θ_{0}$ and $θ_{1}$ that minimize this loss function.

## Gradient Descent

Gradient descent is an optimization algorithm that consists of finding the minimum of a function by iteratively getting closer to it.

In our case, the function we want to minimize is the loss function.

The algorithm works as follows:

1. Initialize the $θ_{0}$ and $θ_{1}$ to 0
2. Calculate the gradient of the loss function, or "what do $θ_{0}$ and $θ_{1}$ miss to be optimal?"
3. Update the $θ_{0}$ and $θ_{1}$ in the opposite direction of the gradient (subtract it)
4. Repeat steps 2 and 3 until the loss function converges

The formula to update the $θ_{0}$ and $θ_{1}$ is:

<!-- $$
θ_{0} = θ_{0} - α \frac{1}{m} \sum_{i=1}^{m} (y_{i} - \hat{y}_{i})
$$ -->

$$
θ_{1} = θ_{1} - α \frac{1}{m} \sum_{i=1}^{m} (y_{i} - \hat{y}_{i})x_{i}
$$

Where:

- $α$ is the learning rate
- $m$ is the number of samples
- $y$ is the actual value
- $\hat{y}$ is the predicted value
- $x$ is the feature

> The learning rate is a hyperparameter that controls how much we update the $θ_{0}$ and $θ_{1}$ at each iteration.
>
> If it's too high, we might overshoot the minimum, if it's too low, we might take too long to converge.

Why do we use these formulas to update the $θ_{0}$ and $θ_{1}$?

- $θ_{0}$ is updated by the average of the errors
- $θ_{1}$ is updated by the average of the errors multiplied by the feature

$θ_{0}$'s update is straightforward, it's just the average of the errors: if your line is ≈ 100 above the actual values, you just need to decrease it by 100.

$θ_{1}$'s update is a bit more complex, given that it's a coefficient. If

# Resources

https://bhatnagar91.medium.com/how-neural-networks-learn-using-gradient-descent-f48c2e4079a6

https://www.youtube.com/watch?v=sDv4f4s2SB8

https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931

https://dilipkumar.medium.com/linear-regression-model-using-gradient-descent-algorithm-50267f55c4ac

https://stats.stackexchange.com/questions/184209/multiple-regression-how-to-calculate-the-predicted-value-after-feature-normali (Plasti Grove's answer)

https://www.youtube.com/watch?v=StHyJm5xcjs

https://medium.com/@mlblogging.k/14-loss-functions-you-can-use-for-regression-b24db8dff987

# Credits

Thanks to [albagar4](https://github.com/albagar4) for the explanation of the coefficient and bias update formulas in the gradient descent.
