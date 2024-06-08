import numpy as np

K = 570
vb = 200
lau = 1.225
F = 0.25
m = 5
g = 9.80665
# g = 35.30394
A = 0.5

def bladePower(v):
    return K / 20 + 3 * ((v/3.6)**2 / vb**2) * K

def parasitePower(v):
    return 0.5 * lau * (v/3.6)**3 * F

def inducedPower(v):
    vi = np.sqrt(-(v/3.6)**2 + np.sqrt((v/3.6)**4 + (m*g)**2 / (lau*A)**2))
    return m * g * vi

def hoveringPower():
    return K + np.sqrt((m * g)**3 / (2 * lau * A))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    v = np.linspace(0, 1000, 1000)
    
    blade_power = [bladePower(i) / 1000 for i in v]
    parasite_power = [parasitePower(i) / 1000 for i in v]
    induced_power = [inducedPower(i) / 1000 for i in v]
    total_power = [blade_power[i] + parasite_power[i] + induced_power[i] for i in range(len(v))]
    
    plt.figure()
    plt.plot(v, blade_power, color='#2F81FC', label='Blade Power')
    plt.plot(v, parasite_power, color='#61FF9E', label='Parasite Power')
    plt.plot(v, induced_power, color='#FF5E5E', label='Induced Power')
    plt.plot(v, total_power, color='#000000', label='Total Power')
    plt.xlabel('Velocity [km/h]')
    plt.ylabel('Power [kW]')
    plt.xlim(0, 70)
    plt.ylim(0, 0.6)
    plt.legend()
    plt.show()
    
    print(round(hoveringPower() / 1000, 2), "kW")