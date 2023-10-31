import numpy as np
import keyboard
import sys    

"""
        __________ x
       /|
      / |
     /  |
   z/   |y
"""

"""
在 move 时，如果直接修改坐标，旋转时的轴会偏移
可以考虑修改 消失点（视点）？或者视平面？

可以这样：
设一个参考点 p, 初始值为 0,0,0 ，然后平移时将这个点跟着移动
旋转曲面的坐标时，先将其坐标 - p， 再以过原点的轴转动，最后再 + p 移动回去
应该可以做出绕p旋转的效果
补充：
其实不需要这么做，因为这里不是移动摄像机而是移动所有坐标，只是显示有问题
检查一下投影面和消失点的问题

映射的时候，若c[3] 大于 映射平面的 z 值。就别映射上去了


考虑再做一个版本 旋转视角 
xy映射后还需要变换吗

"""

numLine = 40

x_max = numLine*3
y_max = numLine*10
x_bias = numLine*1.5
y_bias = numLine*3

theta = np.pi / 16

def project(p0:tuple, n_0:tuple, n_1:tuple, bias:tuple):
    """_summary_

    Args:
        p0 (tuple):     coordinate of the point.
        n_0 (tuple):    normal vectior of the projection plane
        n_1 (tuple):    project vector
        bias (tuple):   reference point coordinate of the projection plane

    Returns:
        _type_: _description_
    """
    x0, y0, z0 = p0[0], p0[1], p0[2]
    m, n, s = n_0[0], n_0[1], n_0[2]
    m1, n1, s1 = n_1[0], n_1[1], n_1[2]
    a, b, c = bias[0], bias[1], bias[2]
    t = -( m*(x0-a) + n*(y0-b) + s*(z0-c) )/(m*m1+n*n1+s*s1)
    x1, y1, z1 = m1*t+x0, n1*t+y0, s1*t+z0
    return (x1, y1, z1)

class Cube(object):
    def __init__(self, row, col, numLine= 20): 
        """_summary_

        Args:
            col (int): 2 * row, number of buffer column.
            numLine (int, optional): _description_. Defaults to 15.
        """
        self.size = (row, col)
        self.numLine = numLine
        
        self.buf = np.zeros((row, col))
        
        
        # no normalization
        # value of |dot| vary at: 0 ~ |illuminant_vector|
        self.illuminant_vector = np.array([0.5773502,0.5773502,0.5773502]) 
        
        refer = numLine //2
        
        self.left_0 = self.__init_coordinate_xz__(-refer, -refer, refer, numLine= numLine, extend= numLine+numLine)
        self.right_0 = self.__init_coordinate_xz__(-refer, refer, refer, numLine= numLine, extend= numLine)
        self.bottom_0 = self.__init_coordinate_yz__(refer, -refer, refer, numLine= numLine, extend= numLine)
        
        self.front_0 = self.__init_coordinate_xy__(-refer, -refer, -refer-numLine-numLine, numLine= numLine, extend= numLine+numLine)
        self.front_1 = self.__init_coordinate_xy__(-refer, refer, -refer-numLine, numLine= numLine, extend= numLine+numLine)
        
        self.left_1 = self.__init_coordinate_xz__(-refer, -refer+3*numLine, refer-3*numLine, numLine= numLine, extend= numLine+numLine)
        self.right_1 = self.__init_coordinate_xz__(-refer, -refer+4*numLine, refer-2*numLine, numLine= numLine, extend= numLine+2*numLine)
        
        
        
        self.coordinate_yz = self.__init_coordinate_yz__(-refer, -refer, refer, numLine= numLine, extend= 20)
        self.coordinate_xz = self.__init_coordinate_xz__(-refer, -refer, refer, numLine= numLine, extend= 20)
        self.coordinate_xy = self.__init_coordinate_xy__(-refer, -refer, refer, numLine= numLine)
        
        self.coordinate_yz_1 = self.__init_coordinate_yz__(refer, -refer, refer, numLine= numLine)
        self.coordinate_xz_1 = self.__init_coordinate_xz__(-refer, refer, refer, numLine= numLine)
        self.coordinate_xy_1 = self.__init_coordinate_xy__(-refer, -refer, -refer, numLine= numLine)
        
        
        
        
        # diagnose
        self.normal_vector = np.array([[-1,0,0],
                                       [0,-1,0],
                                       [0,0,1],
                                       [1,0,0],
                                       [0,1,0],
                                       [0,0,-1]])
        self.normal_vector_ball = np.array([ [c[0]+numLine/2, c[1], c[2] ] for c in self.coordinate_yz])
        self.visual_vector = np.array([0,0,1])
        
        """ ATTENTION
        origin:
            self.disappoint_vec = (0, 0, 3*numLine)
            self.ground = [(0, 0, 1), (0, 0, 0)]
            这样的视角无法很好地模拟实际
        
        投影的失真效果取决于 ground 与 disappear point 之间的距离
        距离越大越接近于平行投影，与人类视角的差距也会变大（可能会穿模，看到另一面墙）
        
        投影在屏幕上的显示是共同决定的，自己调参总结一下
        
        """
        self.disappoint_vec = (0, 0, 0.5*numLine)
        self.ground = [(0, 0, 1), (0, 0, -1.5*numLine)]
        
        
        # self.ground = [(1, 0, 0), (numLine//2, 0, 0)]
        # self.ground = [(0, 1, 0), (0, numLine, 0)]
        
        self.ref_point = np.array((0,0,0))
        self.squares = []
        self.squares.extend([self.left_0, self.right_0, self.front_0, self.front_1, self.left_1, self.right_1])
        self.squares.reverse()
        
        # self.squares = [self.coordinate_yz, self.coordinate_xz, self.coordinate_xy,
        #                 self.coordinate_yz_1, self.coordinate_xz_1, self.coordinate_xy_1]
        # self.temp = []
        # for square in self.squares[:]:
        #     self.temp.append( square + (0, 1.25*numLine, -1.25*numLine) )        
        # self.squares.extend(self.temp)
        
        
        self.lightMap = [' ', '.', '*', '%']
        self.lightMap = " .-:;=+*#%@@@@@@"
        self.lightMap_ball = " .--:;=+**##%@@"
        
        self.movementMap = {
            'w':(0,0,numLine//10),
            's':(0,0,-numLine//10),
            'a':(0,numLine//10,0),
            'd':(0,-numLine//10,0),
            'space':(numLine//10,0,0),
            'shift':(-numLine//10,0,0),
        }
        pass
    
    def __init_yz__(self, x_0, y_0, z_0, numLine):
        return np.array( [[(x_0, y_0 + i, z_0 + j) for i in range(-numLine//2 ,numLine//2)] for j in range(-numLine//2, numLine//2)],
                            dtype= np.float64 ).reshape(numLine*numLine, 3)
    def __init_xz__(self, x_0, y_0, z_0, numLine):
        return np.array( [[(x_0 + i, y_0, z_0 + j) for i in range(-numLine//2 ,numLine//2)] for j in range(-numLine//2, numLine//2)],
                            dtype= np.float64 ).reshape(numLine*numLine, 3)
    def __init_xy__(self, x_0, y_0, z_0, numLine):
        return np.array( [[(x_0 + i, y_0 + j, z_0) for i in range(-numLine//2 ,numLine//2)] for j in range(-numLine//2, numLine//2)],
                            dtype= np.float64 ).reshape(numLine*numLine, 3)
    
    def __init_coordinate_yz__(self, x_0, y_0, z_0, numLine, extend = 0):
        return np.array( [[(x_0, y_0 + i, z_0 - j) for i in range(numLine)] for j in range(numLine + extend)],
                            dtype= np.int16 ).reshape(numLine*(numLine + extend), 3)
        
    def __init_coordinate_xz__(self, x_0, y_0, z_0, numLine, extend = 0):
        return np.array( [[(x_0 + i, y_0, z_0 - j) for i in range(numLine)] for j in range(numLine + extend)],
                            dtype= np.int16 ).reshape(numLine*(numLine + extend), 3)
        
    def __init_coordinate_xy__(self, x_0, y_0, z_0, numLine, extend = 0):
        return np.array( [[(x_0 + j, y_0 + i, z_0) for i in range(numLine + extend)] for j in range(numLine)],
                            dtype= np.int16 ).reshape(numLine*(numLine + extend), 3)
    
    def __rotate__(theta= 0, axis= "x"):
        def rotate_x(theta):
            return np.array([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)] ])
        def rotate_y(theta):
            return np.array([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)] ])
        AxisMap = {"x":rotate_x, "y":rotate_y}
        return AxisMap[axis](theta)
    
    def rotate(self, theta= 0, axis= "x"):
        rotate = Cube.__rotate__(theta, axis)
        for idx, square in enumerate(self.squares):
            self.squares[idx] = np.dot(square, rotate)
        self.normal_vector = np.dot(self.normal_vector, rotate)
        self.normal_vector_ball = np.dot(self.normal_vector_ball, rotate)
        
    def rotate_ref(self, theta= 0, axis= "x"):
        rotate = Cube.__rotate__(theta, axis)
        for idx, square in enumerate(self.squares):
            self.squares[idx] = np.dot(square, rotate)
            # square = square - self.ref_point
            # self.squares[idx] = np.dot(square, rotate) + self.ref_point
        self.normal_vector = np.dot(self.normal_vector, rotate)
        self.normal_vector_ball = np.dot(self.normal_vector_ball, rotate)
    
    def move(self, direction):
        # bias = (0,0,0)
        # if direction == 'w':
        #     bias = (0,0,2)
        # elif direction == 's':
        #     bias = (0,0,-2)
        # elif direction == 'a':
        #     bias = (0,2,0)
        # elif direction == 'd':
        #     bias = (0,-2,0)
        # elif direction == 'space':
        #     bias = (2,0,0)
        # elif direction == 'shift':
        #     bias = (-2,0,0)
        
        bias = self.movementMap[direction]
        self.ref_point += bias
        # self.ground += bias
        for i, square in enumerate(self.squares):
                self.squares[i] = square + bias
        
        
        pass
    
    def rotate_light(self, theta= 0, axis= "x"):
        rotate = Cube.__rotate__(theta, axis)
        self.illuminant_vector = np.dot(self.illuminant_vector, rotate)
        
    def reset(self):
        self.coordinate_xy = self.__init_coordinate_xy__(-10, -10, 10, 20)
        self.coordinate_yz = self.__init_coordinate_yz__(-10, -10, 10, 20)
        self.coordinate_xz = self.__init_coordinate_xz__(-10, -10, 10, 20)
    
    def writeBuf(self, x_bias= 30, y_bias= 40):
        row = self.size[0]
        col = self.size[1]
        # normal vector of each square dot the vector of our sight
        dot_product = np.dot(self.normal_vector, self.visual_vector)
        # normal vector of each square dot the vector of light
        light_product = np.dot(self.normal_vector, self.illuminant_vector) * 5 + 6
        self.light = light_product
        
        def square2buf(cube: Cube, square, light, x_bias, y_bias):
            for c in square:
                # 实质是计算待投影坐标与消失点的距离, 越过消失点的点不要投影
                if c[2] > self.disappoint_vec[2]:
                    continue
                # if c[2] > self.ground[1][2]:
                #     continue
                
                project_vec = - c + self.disappoint_vec
                c_ = project(c, self.ground[0], project_vec, self.ground[1])
                try:
                    x = int(c_[0]+x_bias)
                    y = int(c_[1]*1.5+y_bias)
                except (OverflowError, ValueError):
                    continue
                # todo optimize
                if x in range(0,row) and y in range(0,col):
                    cube.buf[x][y] = light
                # if (x >= row or y >= col):
                #     return
                # else:
                #     cube.buf[x][y] = light
        for i, square in enumerate(self.squares[:6]):
            # 依照投影向量与各个面法向量的点积, 决定当前应该显示哪个面
            square2buf(self, square, i+1, x_bias, y_bias)  

        # for i, square in enumerate(self.squares[:6]):
        #     # 依照投影向量与各个面法向量的点积, 决定当前应该显示哪个面
        #     if dot_product[i] > 0:
        #         square2buf(self, square, light_product[i], x_bias, y_bias)  
        # for i, square in enumerate(self.squares[6:]):
        #     # 依照投影向量与各个面法向量的点积, 决定当前应该显示哪个面
        #     square2buf(self, square, light_product[i], x_bias, y_bias)  
        
        
        
    
    def show(self):
        def fresh():
            import os
            os.system("cls")
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.flush()
        fresh()
        for row in self.buf:
            for column in row:
                print(self.lightMap[int(column)], end='');
            print('')
        # refresh buf
        self.buf = np.zeros_like(self.buf)
        print(self.ref_point)
        # print(self.normal_vector)
        # print(self.testnormal)
        # print(self.testlight)
        # print(self.squares[0])

class KeyHandler(object):
    def hadler_move(self, event):
        if event.name not in ['w','a','s','d','space','shift']:
            return
        cube.move(event.name)
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        
    # cube
    def handler_w(self, event):
        # cube.rotate(theta, axis= "y")
        cube.move(event.name)
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    def handler_s(self, event):
        # cube.rotate(-theta, axis= "y")
        cube.move(event.name)
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    def handler_d(self, event):
        # cube.rotate(theta, axis= "x")
        cube.move(event.name)
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    def handler_a(self, event):
        cube.move(event.name)
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    
    # light
    def handler_up(self, event):
        cube.rotate_ref(theta, axis= "y")
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    def handler_down(self, event):
        cube.rotate_ref(-theta, axis= "y")
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    def handler_left(self, event):
        cube.rotate_ref(-theta, axis= "x")
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    def handler_right(self, event):
        cube.rotate_ref(theta, axis= "x")
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    
    def handler_reset(event):
        cube.reset()
        cube.writeBuf()
        cube.show()
        pass

cube = Cube(x_max, y_max, numLine)
handler = KeyHandler()

if __name__ == "__main__":
    # cube.__init_coordinate__(20, 10, 10)
    # cube.writeBuf(theta= np.pi/6, axis= "x")d
    # keyboard.on_press_key("w", handler.handler_w)
    # keyboard.on_press_key("a", handler.handler_a)
    # keyboard.on_press_key("s", handler.handler_s)
    # keyboard.on_press_key("d", handler.handler_d)
    
    keyboard.on_press_key("up", handler.handler_up)
    keyboard.on_press_key("down", handler.handler_down)
    keyboard.on_press_key("left", handler.handler_left)
    keyboard.on_press_key("right", handler.handler_right)
    
    keyboard.on_press(handler.hadler_move)
    keyboard.wait("esc")
    # cube.rotate(theta= np.pi/6, axis= "y")
    # cube.rotate(theta= np.pi/6, axis= "y")
    # cube.rotate(theta= np.pi/4, axis= "x")
    
    pass