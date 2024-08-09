def read_thetas():
	try:
		thetas = open("thetas", "r")
	except:
		print("Error: Could not open file")
		exit()

	theta = thetas.read().split(',')

	try:
		theta = [float(i) for i in theta]
		thetas.close()
	except:
		print("Error: Could not convert theta values to float")
		exit()

	return theta

def estimate_price(mileage):
	theta = read_thetas()

	return theta[0] + (theta[1] * mileage)

def main():
	mileage = input("Enter mileage: ")
	try:
		mileage = float(mileage)
	except:
		print("Error: Could not convert mileage to float")
		exit()

	price = estimate_price(mileage)
	print("Estimated price: ", price)

if __name__ == "__main__":
	main()
