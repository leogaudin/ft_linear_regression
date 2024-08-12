train.py
```python
from estimate import estimate_price
from utils import read_data, plot_final_state, plot_loss
import matplotlib.pyplot as plt

class lin_reg:
	def __init__(self, file='data.csv'):
		# Init hyperparameters
		self.learning_rate = 0.1
		self.theta0 = 0
		self.theta1 = 0

		# Init data
		self.raw_data = read_data(file)
		self.raw_mileages = [row[0] for row in self.raw_data]
		self.raw_prices	= [row[1] for row in self.raw_data]

		# Init metas
		self.mean_mileage = sum(self.raw_mileages) / len(self.raw_mileages)
		self.std_dev_mileage = (sum([(mileage - self.mean_mileage) ** 2 for mileage in self.raw_mileages]) / len(self.raw_mileages)) ** 0.5
		self.m = len(self.raw_data)

		# Scale data
		self.mileages = [(raw_mileage - self.mean_mileage) / self.std_dev_mileage for raw_mileage in self.raw_mileages]
		self.prices = self.raw_prices

		# Accumulators
		self.loss_acc = []


	""" Prints the current theta values and loss
	"""
	def print_state(self, epoch):
		print(
			'Epoch\t{}\nθ0\t{}\nθ1\t{}\nLoss\t{}\n'
				.format(
					epoch,
					(self.theta0 - self.theta1 * self.mean_mileage / self.std_dev_mileage),
					(self.theta1 / self.std_dev_mileage),
					self.loss_acc[-1]
				)
		)


	""" Calculates the errors for the current theta values

	Returns:
		float -- t0 error
		float -- t1 error
		float -- total loss
	"""
	def calculate_errors(self):
		t0_error = 0
		t1_error = 0
		total_loss = 0

		for i in range(self.m - 1):
			prediction = estimate_price(self.theta0, self.theta1, self.mileages[i])
			error = prediction - self.prices[i]
			t0_error += error
			t1_error += error * self.mileages[i]
			total_loss += abs(error)

		total_loss /= self.m
		return t0_error, t1_error, total_loss


	""" Trains the model by repeatedly making hypothesis and updating thetas in the appropriate direction
	"""
	def train(self):
		max_epoch = 1000
		weighted_learning_rate = self.learning_rate / self.m

		# # Plot evolution of hypothesis
		# plt.figure()
		# plt.ion()
		# plt.scatter(self.mileages, self.prices, color='blue')

		# Main loop
		for epoch in range(max_epoch + 1):
			t0_error, t1_error, loss = self.calculate_errors()
			self.theta0 -= weighted_learning_rate * t0_error
			self.theta1 -= weighted_learning_rate * t1_error
			self.loss_acc.append(loss)

			if len(self.loss_acc) > 1 and round(self.loss_acc[-1], 7) == round(self.loss_acc[-2], 7):
				break

			if (epoch % (max_epoch / 1000)) == 0:
				self.print_state(epoch)

		# 	# Plot hypothesis
		# 	plt.plot(
		# 		[min(self.mileages), max(self.mileages)],
		# 		[estimate_price(self.theta0, self.theta1, min(self.mileages)), estimate_price(self.theta0, self.theta1, max(self.mileages))],
		# 		'r'
		# 	)
		# 	plt.pause(0.1)

		# plt.ioff()

		# Unscale thetas
		self.theta0 -= (self.theta1 * self.mean_mileage / self.std_dev_mileage)
		self.theta1 /= self.std_dev_mileage

		plot_final_state(self)
		plot_loss(self)


	""" Stores the theta values in the thetas file
	"""
	def store_thetas(self):
		try:
			output = open('thetas', 'w')
			output.write(str(self.theta0) + ',' + str(self.theta1))
			output.close()
		except:
			print('Error during theta storing')
			exit()

""" Main function """
def main():
	lr = lin_reg()
	lr.train()
	lr.store_thetas()
	plt.show()

if __name__ == '__main__':
	main()
```

---

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

$$
θ_{0} = θ_{0} - α \frac{1}{m} \sum_{i=1}^{m} (y_{i} - \hat{y}_{i})
$$

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
