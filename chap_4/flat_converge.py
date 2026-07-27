gamma = 0.9
V = {"Sherlock": 0, "Avenger": 0, "Heat": 0}

print(f"{'Itér.':<8} {'V(Sherlock)':<16} {'V(Avenger)':<16} {'V(Heat)':<16}")
print("-" * 56)
print(f"{'0':<8} {V['Sherlock']:<16.4f} {V['Avenger']:<16.4f} {V['Heat']:<16.4f}")

for i in range(1, 6):
    new_V = {}
    new_V["Heat"]     = 0
    new_V["Avenger"]  = 8 + gamma * V["Heat"]
    new_V["Sherlock"] = 2 + gamma * V["Avenger"]
    V = new_V
    print(f"{i:<8} {V['Sherlock']:<16.4f} {V['Avenger']:<16.4f} {V['Heat']:<16.4f}")