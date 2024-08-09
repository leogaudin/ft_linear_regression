""" Reads the values from thetas file

Returns:
	list -- list of theta values
"""
def read_thetas():
	try:
		thetas = open("thetas", "r")
	except:
		return [0, 0]

	theta = thetas.read().split(',')

	try:
		theta = [float(i) for i in theta]
		thetas.close()
	except:
		return [0, 0]

	return theta


""" Estimates the price of a car given its mileage

Arguments:
	mileage {float} -- mileage of the car

Returns:
	float -- estimated price of the car
"""
def estimate_price(theta0, theta1, mileage):
	return theta0 + (theta1 * mileage)


""" Main function """
def main():
	mileage = input("Enter mileage: ")
	try:
		mileage = float(mileage)
	except:
		print("Error: Could not convert mileage to float")
		exit()

	thetas = read_thetas()
	price = estimate_price(thetas[0], thetas[1], mileage)
	print("Estimated price: ", price)

if __name__ == "__main__":
	main()
