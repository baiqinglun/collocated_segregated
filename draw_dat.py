'''绘图功能'''
import matplotlib.pyplot as plt

# 读取dat文件
with open(file='output/center_temp_x_center.dat',mode='r', encoding="utf-8") as file:
    lines = file.readlines()

# 提取数据
x_values = []
y1_values = []
y2_values = []

for line in lines:
    values = line.split()
    x_values.append(float(values[0]))
    y1_values.append(float(values[1]))
    y2_values.append(float(values[2]))

# 绘制图表
plt.plot(x_values, y1_values, label='central solution')
plt.plot(x_values, y2_values, label='exact solution')

# 添加标签和图例
plt.xlabel('PeL')
plt.ylabel('Y轴')
plt.legend()

# 显示图表
plt.show()
