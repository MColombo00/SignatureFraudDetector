import statistics as s
data = []

with open("CornerDetect\\data\\org_data_cords.csv") as f:
    for i, line in enumerate(f, 1):
        points = [p for p in line.strip().split("|") if p]
        data.append(len(points))
        print(f"Image {i}: {len(points)} points")

print(s.mean(data))