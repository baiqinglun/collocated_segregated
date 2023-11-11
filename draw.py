import matplotlib.pyplot as plt

"""
@name: DrawCurves
@description: 绘制残差曲线图
@variable:
    ax:轴
    time:[]
    residuals:[[]]
@function: 
    draw()：绘图
"""
class DrawCurves:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel('time')
        self.ax.set_ylabel('residual')
        self.ax.set_title('residual-time')
        plt.ion()
        self.time = []
        self.residuals = [[]]

        self.is_show = True

    def draw(self,x,ys):
        self.time.append(x)
        for index,y in enumerate(ys):
            self.residuals[index].append(y)
        for index,residual in enumerate(self.residuals):
            line = self.ax.plot(self.time, residual, '-g', marker='*')[0]
        plt.pause(0.01)