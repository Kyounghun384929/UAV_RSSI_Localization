import numpy as np

class AP:
    def __init__(self, x, y, distance):
        self.x = x
        self.y = y
        self.distance = distance

class Multilateration:
    def __init__(self, APs):
        self.APs = APs
    
    def calcUserLocation(self):
        A = []  # Initialize list to store coefficients of linear equations
        b = []  # Initialize list to store constants of linear equations
        for i in range(len(self.APs)):
            if i < len(self.APs) - 1:
                x_diff = 2 * (self.APs[i+1].x - self.APs[i].x)  # Calculate x difference
                y_diff = 2 * (self.APs[i+1].y - self.APs[i].y)  # Calculate y difference
                dist_diff = self.APs[i].distance**2 - self.APs[i+1].distance**2 \
                            - self.APs[i].x**2 + self.APs[i+1].x**2 \
                            - self.APs[i].y**2 + self.APs[i+1].y**2  # Calculate distance difference
                A.append([x_diff, y_diff])  # Append coefficients to A matrix
                b.append([dist_diff])  # Append constants to b matrix
        
        A = np.array(A)  # Convert A to numpy array
        b = np.array(b)  # Convert b to numpy array
        
        user_location = np.linalg.lstsq(A, b, rcond=None)[0]  # Solve linear equations to find user location
        return user_location.flatten()  # Return user location as flattened array
    
if __name__ == "__main__":
    ap1 = AP(14, 64, np.sqrt(32))  # Create AP object with coordinates and distance
    ap2 = AP(123, 24, np.sqrt(32))  # Create AP object with coordinates and distance
    ap3 = AP(81, 12, 4)  # Create AP object with coordinates and distance
    ap4 = AP(42, 48, 6)  # Create additional AP object with coordinates and distance
    
    multilat = Multilateration([ap1, ap2, ap3, ap4])  # Create Multilateration object with APs
    x, y = multilat.calcUserLocation()  # Calculate user location
    print(f"Estimated User Location: x={x}, y={y}")  # Print estimated user location

    import matplotlib.pyplot as plt  # Import matplotlib for visualization
    
    plt.figure()  # Create a new figure
    plt.plot(ap1.x, ap1.y, 'o')  # Plot AP1
    plt.plot(ap2.x, ap2.y,'o')  # Plot AP2
    plt.plot(ap3.x, ap3.y,'o')  # Plot AP3
    plt.plot(ap4.x, ap4.y,'o')  # Plot AP4
    plt.plot(x, y, 'o', markersize=30, label='Estimated Location')  # Plot estimated user location
    plt.legend()  # Show legend
    plt.show()  # Display the plot
