from estimate import estimate_price
import matplotlib.pyplot as plt

def normalize(element, list):
	return (element - min(list)) / (max(list) - min(list))

def denormalize(element, list):
	return element * (max(list) - min(list)) + min(list)

def read_data():
		try:
			data = open("data.csv", "r")
		except:
			print("Error: Could not open file")
			exit()

		data = data.read().split('\n')
		del data[0]
		del data[-1]
		data = [line.split(',') for line in data]

		try:
			data = [[float(element) for element in line] for line in data]
		except:
			print("Error: Could not convert data values to float")
			exit()

		return data

class lin_reg:
	def __init__(self):
		self.learning_rate = 0.75
		self.theta0 = 0
		self.theta1 = 0
		self.raw_data = read_data()
		self.raw_mileages = [row[0] for row in self.raw_data]
		self.raw_prices	= [row[1] for row in self.raw_data]
		self.m = len(self.raw_data)
		self.normalize_data()


	def get_min_max_prices(self):
		return min(self.raw_prices), max(self.raw_prices)


	def get_min_max_mileages(self):
		return min(self.raw_mileages), max(self.raw_mileages)


	def print_meta(self):
		print("Min max mileages: ", self.get_min_max_mileages())
		print("Min max prices: ", self.get_min_max_prices())
		print("Theta0: ", self.theta0)
		print("Theta1: ", self.theta1)
		print("Learning rate: ", self.learning_rate)
		print("Number of samples: ", self.m)


	def normalize_data(self):
		self.mileages = []
		self.prices = []

		for line in self.raw_data:
			self.mileages.append(normalize(line[0], self.raw_mileages))
			self.prices.append(normalize(line[1], self.raw_prices))


	def calculate_gradients(self):
		sum0 = 0
		sum1 = 0

		for i in range(self.m - 1):
			sum0 += estimate_price(self.theta0, self.theta1, self.mileages[i]) - self.prices[i]
			sum1 += (estimate_price(self.theta0, self.theta1, self.mileages[i]) - self.prices[i]) * self.mileages[i]

		return [
			self.learning_rate * sum0 / self.m,
			self.learning_rate * sum1 / self.m
		]

	def loss(self):
		error = 0
		for line in self.raw_data:
			error += abs(normalize(line[1], self.raw_prices) - estimate_price(
																	self.theta0,
																	self.theta1,
																	normalize(line[0], self.raw_mileages)
																))

		error /= self.m - 1
		error *= self.learning_rate
		return error

	def train(self):
		max_epoch = 1000

		for epoch in range(max_epoch + 1):
			gradients = self.calculate_gradients()
			self.theta0 -= gradients[0]
			self.theta1 -= gradients[1]
			error = self.loss()

			if (epoch % (max_epoch / 10)) == 0:
				print("Epoch {}\t\tt0 {} - t1 {} - error {}".format(epoch, self.theta0, self.theta1, error))

		print("Training finished")


def display_normalized_data(lr):
	normalized_mileages = [normalize(mileage, lr.raw_mileages) for mileage in lr.raw_mileages]
	normalized_prices = [normalize(price, lr.raw_prices) for price in lr.raw_prices]
	plt.plot(normalized_mileages, normalized_prices, 'bo')

	min_mileage, max_mileage = lr.get_min_max_mileages()
	normalize_min_mileage = normalize(min_mileage, lr.raw_mileages)
	normalize_max_mileage = normalize(max_mileage, lr.raw_mileages)
	min_result = estimate_price(lr.theta0, lr.theta1, normalize_min_mileage)
	max_result = estimate_price(lr.theta0, lr.theta1, normalize_max_mileage)
	plt.plot(
		[normalize_min_mileage, normalize_max_mileage],
		[min_result, max_result],
		'r'
	)
	plt.show()


def main():
	lr = lin_reg()

	lr.train()
	lr.print_meta()

	display_normalized_data(lr)

if __name__ == "__main__":
	main()
