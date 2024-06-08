import numpy as np

class AP:
    def __init__(self, x, y, distance):
        self.x = x
        self.y = y
        self.distance = distance

class Trilateration:
    def __init__(self, AP1, AP2, AP3):
        self.AP1 = AP1
        self.AP2 = AP2
        self.AP3 = AP3
    
    def calcUserLocation(self):
        A = 2 * (self.AP2.x - self.AP1.x)
        B = 2 * (self.AP2.y - self.AP1.y)
        C = self.AP1.distance**2 - self.AP2.distance**2 - self.AP1.x**2 + self.AP2.x**2 - self.AP1.y**2 + self.AP2.y**2
        D = 2 * (self.AP3.x - self.AP2.x)
        E = 2 * (self.AP3.y - self.AP2.y)
        F = self.AP2.distance**2 - self.AP3.distance**2 - self.AP2.x**2 + self.AP3.x**2 - self.AP2.y**2 + self.AP3.y**2
        
        user_x = ( (F * B) - (E * C) ) / ( (B * D) - (E * A))
        user_y = ( (F * A) - (D * C) ) / ( (A * E) - (D * B))
        return user_x, user_y
    
if __name__ == "__main__":
    ap1 = AP(14, 64, np.sqrt(32))
    ap2 = AP(123, 24, np.sqrt(32))
    ap3 = AP(81, 12, 4)

    tril = Trilateration(ap1, ap2, ap3)
    x, y = tril.calcUserLocation()
    print(x, y)