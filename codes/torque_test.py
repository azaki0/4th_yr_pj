import math

g = 9.81
#1in = 0.0254m

m1 = [i * 0.01 for i in range(21)]
m2 = [i * 0.01 for i in range(21)]

#for 25:1
#gear_torque = 12.5 * 0.7

#for 64:1
gear_torque = 33.28 * 0.75

max_mass = gear_torque/g*6*0.0254
entries = []

for i in m1:
    for j in m2:
        mass = i+3*j
        if mass< max_mass:
            entries.append((i, j, mass))

x = [entry[0] for entry in entries]
x_1 = [entry[1] for entry in entries]
y = [entry[2] for entry in entries]

entries = [entry for entry in entries if entry[0] > 0 and entry[1] > 0]

max_val = max(entries, key=lambda x: x[2])[2]
max_possible_mass = [entry for entry in entries if math.isclose(entry[2], max_val, rel_tol=1e-9)]

print("mass 1, mass 2, total mass")
print('------------------------------')
print('\n'.join([f"{entry[0]:.2f}, {entry[1]:.2f}, {entry[2]:.2f}" for entry in entries]))
print('------------------------------')
for mass in max_possible_mass:
    print(f"max mass(g): {mass[0]*1000}, {mass[1]*1000}, {1000*mass[2]:.2f}, {gear_torque} N-m")