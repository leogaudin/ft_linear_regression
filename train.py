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
