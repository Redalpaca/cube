import numpy as np
import keyboard
import sys    

numLine = 40
x_max = numLine*2
y_max = numLine*4
x_bias = numLine
y_bias = numLine*2
theta = np.pi / 16

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
        
        refer = numLine // 2
        self.coordinate_yz = self.__init_coordinate_yz_ball__(-refer, -refer, refer, numLine= numLine)
        self.heart, self.normal_vector = self.__init_heart__()
        
        # diagnose
        self.visual_vector = np.array([0,0,1])
        
        self.squares = [self.heart]
        # self.squares = [self.coordinate_yz, self.coordinate_xz, self.coordinate_xy]
        self.lightMap = [' ', '.', '*', '%']
        self.lightMap = " .-:;=+*##%@@@@@@"
        # self.lightMap = [" ", "\033[31m.\033[0m","\033[32m.\033[0m","\033[32m.\033[0m","\033[33m.\033[0m","\033[33m.\033[0m","\033[34m.\033[0m","\033[34m.\033[0m","\033[35m.\033[0m","\033[35m.\033[0m","\033[36m.\033[0m","\033[36m.\033[0m","\033[36m.\033[0m","\033[36m.\033[0m","\033[36m.\033[0m"]
        pass
    
    def __init_coordinate_yz__(self, x_0, y_0, z_0, numLine):
        return np.array( [[(x_0, y_0 + i, z_0 - j) for i in range(numLine)] for j in range(numLine)],
                            dtype= np.int16 ).reshape(numLine*numLine, 3)
        
    def __init_coordinate_yz_ball__(self, x_0, y_0, z_0, numLine):
        def calc_x(x, y, z):
            # y = y + numLine // 2
            # z = z - numLine // 2
            if y**2 + z**2 >= numLine*numLine / 4:
                return x
            else:
                return x - (numLine**2 / 4 - y**2 - z**2)**0.5
        return np.array( [[(calc_x(x_0, y_0+i, z_0-j), y_0 + i, z_0 - j) for i in range(numLine)] for j in range(numLine)],
                            dtype= np.float32 ).reshape(numLine*numLine, 3)
        # return np.array( [[(x_0, y_0 + i, z_0 - j) for i in range(numLine)] for j in range(numLine)],
        #                            dtype= np.int32 ).reshape(numLine*numLine, 3)
    
    # f(x,y) = 0 => f(+- sqrt(x^2+z^2), y) = 0
    # def __init_heart_flat__(self, r):
    #     line = np.array([(x, (x-2.4142*r), 0) for x in range(0, int(1.7071*r))])
    #     circle = [(x, (r*r-(x-r)*(x-r))**0.5, 0) for x in range(0, 2*r)]
    #     circle.extend([(x, -(r*r-(x-r)*(x-r))**0.5, 0) for x in range(int(1.7071*r), 2*r)])
    #     circle = np.array(circle)
    #     return (circle, line)
    
    # 参数方程的梯度
    # (dy/dt, -dx/dt)
    def __init_heart_flat__(self, param_x= 16, param_y= np.array((13,5,2,1))):
        def _x_(t, p=16):
            return p*np.sin(t)**3
        def _y_(t, p=(13,5,2,1)):
            return p[0]*np.cos(t)-p[1]*np.cos(2*t)-p[2]*np.cos(3*t)-p[3]*np.cos(4*t)
        def _dx_(t, p=16):
            return 3*p*np.cos(t)*np.sin(t)**2
        def _dy_(t, p=(13,5,2,1)):
            return -p[0]*np.sin(t)+ 2*p[1]*np.sin(2*t)+ 3*p[2]*np.sin(3*t)+ 4*p[3]*np.sin(4*t)
        # param
        iter_ = 150
        param_x = 16 * 2
        param_y = np.array((13,5,2,1)) * 2
        
        theta = [ i*np.pi/iter_*2 for i in range(iter_)]
        curve = np.array([(_x_(t, param_x), _y_(t, param_y), 0) for t in theta])
        norm_vec = []
        for t in theta:
            v = ( _dy_(t, param_y), -_dx_(t, param_x), 0)
            norm = (v[0]**2 + v[1]**2) # np.linalg.norm(v, 2)
            if norm != 0:
                norm_vec.append(v/norm**0.5)
            else:
                norm_vec.append(v)
        norm_vec = np.array(norm_vec)
        return (curve, norm_vec)
    
    def __init_heart__(self):
        iter_ = 100
        theta = [ i*np.pi/iter_ for i in range(iter_)]
        curve_origin, norm_vec_origin = self.__init_heart_flat__()
        curve = curve_origin.copy()
        norm_vec = norm_vec_origin.copy()
        for t in theta:
            next_c = np.dot(curve_origin, Cube.__rotate__(t, "y"))
            curve = np.concatenate((curve, next_c))
            next_v = np.dot(norm_vec_origin, Cube.__rotate__(t, "y"))
            norm_vec = np.concatenate((norm_vec, next_v))
        return (curve, norm_vec)
    
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
        
    def rotate_light(self, theta= 0, axis= "x"):
        rotate = Cube.__rotate__(theta, axis)
        self.illuminant_vector = np.dot(self.illuminant_vector, rotate)
        pass
    
    
    def reset(self):
        pass
    
    def writeBuf(self, x_bias= 30, y_bias= 40):
        row = self.size[0]
        col = self.size[1]
        # normal vector of each square dot the vector of our sight
        dot_product = np.dot(self.normal_vector, self.visual_vector)
        # normal vector of each square dot the vector of light
        light_product = np.dot(self.normal_vector, self.illuminant_vector) * 5 + 6
        self.light = light_product
        
        def square2buf(cube: Cube, square, light, x_bias, y_bias):
            for i, c in enumerate(square):
                if self.normal_vector[i][2] > 0:
                    cube.buf[int(c[0]+x_bias)%row, int(2*c[1]+y_bias)%col] = light_product[i]
                
        for i, square in enumerate(self.squares):
            # 依照投影向量与各个面法向量的点积, 决定当前应该显示哪个面
            square2buf(self, square, light_product[i], x_bias, y_bias)  
        
        # self.writeBuf_add1()
            
    def writeBuf_add1(self):
        global x_max, y_max
        for i in range(x_bias, x_max):
            for j in range(y_max-10, y_max):
                self.buf[i][j] = 1
            pass
        pass
    
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
        # print(self.normal_vector)
        # print(self.squares[0])

class KeyHandler(object):
    # cube
    def handler_w(self, event):
        cube.rotate(theta, axis= "y")
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    def handler_s(self, event):
        cube.rotate(-theta, axis= "y")
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    def handler_d(self, event):
        cube.rotate(theta, axis= "x")
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    def handler_a(self, event):
        cube.rotate(-theta, axis= "x")
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    
    # light
    def handler_up(self, event):
        cube.rotate_light(2*theta, axis= "y")
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    def handler_down(self, event):
        cube.rotate_light(-2*theta, axis= "y")
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    def handler_left(self, event):
        cube.rotate_light(-2*theta, axis= "x")
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        pass
    def handler_right(self, event):
        cube.rotate_light(2*theta, axis= "x")
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
    keyboard.on_press_key("w", handler.handler_w)
    keyboard.on_press_key("a", handler.handler_a)
    keyboard.on_press_key("s", handler.handler_s)
    keyboard.on_press_key("d", handler.handler_d)
    
    keyboard.on_press_key("up", handler.handler_up)
    keyboard.on_press_key("down", handler.handler_down)
    keyboard.on_press_key("left", handler.handler_left)
    keyboard.on_press_key("right", handler.handler_right)
    keyboard.on_press_key(" ", handler.handler_reset)
    keyboard.wait("esc")
    # cube.rotate(theta= np.pi/6, axis= "y")
    # cube.rotate(theta= np.pi/6, axis= "y")
    # cube.rotate(theta= np.pi/4, axis= "x")
    
    pass