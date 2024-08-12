from estimate import estimate_price
import matplotlib.pyplot as plt

""" Reads the values from data.csv file

Returns:
	list -- list of data values
"""
def read_data(file):
		try:
			data = open(file, 'r')
			data = data.read().split('\n')
			del data[0]
			del data[-1]
			data = [line.split(',') for line in data]
			data = [[float(element) for element in line] for line in data]
		except:
			print('Error during data reading')
			exit()

		return data


""" Displays the data as dots and the hypothesis as a line

Arguments:
	lr {lin_reg} -- linear regression object
"""
def plot_final_state(lr):
	plt.figure()
	plt.title('Data and hypothesis')
	plt.xlabel('Mileage')
	plt.ylabel('Price')
	plt.scatter(lr.raw_mileages, lr.raw_prices, color='blue')
	plt.plot(
		[min(lr.raw_mileages), max(lr.raw_mileages)],
		[estimate_price(lr.theta0, lr.theta1, min(lr.raw_mileages)), estimate_price(lr.theta0, lr.theta1, max(lr.raw_mileages))],
		'r'
	)

""" Displays the evolution of the loss over time

Arguments:
	lr {lin_reg} -- linear regression object
"""
def plot_loss(lr):
	plt.figure()
	plt.title('Loss over time')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.plot([i for i in range(len(lr.loss_acc))], lr.loss_acc, 'r')
