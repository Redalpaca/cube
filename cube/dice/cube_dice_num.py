import numpy as np
import keyboard
import sys    

numLine = 20
x_max = numLine*2
y_max = numLine*4
x_bias = numLine
y_bias = numLine*2

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
        
        # diagnose
        self.normal_vector = np.array([[-1,0,0],
                                       [0,-1,0],
                                       [0,0,1],
                                       [1,0,0],
                                       [0,1,0],
                                       [0,0,-1]])
        self.visual_vector = np.array([0,0,1])
        
        refer = numLine // 2
        self.coordinate_yz = self.__init_coordinate_yz__(-refer, -refer, refer, numLine= numLine)
        self.coordinate_xz = self.__init_coordinate_xz__(-refer, -refer, refer, numLine= numLine)
        self.coordinate_xy = self.__init_coordinate_xy__(-refer, -refer, refer, numLine= numLine)
        
        self.coordinate_yz_1 = self.__init_coordinate_yz__(refer, -refer, refer, numLine= numLine)
        self.coordinate_xz_1 = self.__init_coordinate_xz__(-refer, refer, refer, numLine= numLine)
        self.coordinate_xy_1 = self.__init_coordinate_xy__(-refer, -refer, -refer, numLine= numLine)
        
        self.squares = [self.coordinate_yz, self.coordinate_xz, self.coordinate_xy,
                        self.coordinate_yz_1, self.coordinate_xz_1, self.coordinate_xy_1]
        # self.squares = [self.coordinate_yz, self.coordinate_xz, self.coordinate_xy]
        self.lightMap = [' ', '.', '1','2','3','6','5','4']
        pass
    
    def __init_coordinate_yz__(self, x_0, y_0, z_0, numLine):
        return np.array([
                [(x_0, y_0, z_0 - i) for i in range(numLine)],
                [(x_0, -y_0, z_0 - i) for i in range(numLine)],
                [(x_0, y_0 + i, z_0) for i in range(numLine)],
                [(x_0, y_0 + i, -z_0) for i in range(numLine)]
                ]).reshape(numLine*4, 3)
        # return np.array( [[(x_0, y_0 + i, z_0 - j) for i in range(numLine)] for j in range(numLine)],
        #                             dtype= np.int32 ).reshape(numLine*numLine, 3)
    
    def __init_coordinate_xz__(self, x_0, y_0, z_0, numLine):
        return np.array([
                [(x_0 + i, y_0, z_0) for i in range(numLine)],
                [(x_0 + i, y_0, -z_0) for i in range(numLine)],
                [(x_0, y_0, z_0 - i) for i in range(numLine)],
                [(-x_0, y_0, z_0 - i) for i in range(numLine)]
                ]).reshape(numLine*4, 3)
        # return np.array( [[(x_0 + i, y_0, z_0 - j) for i in range(numLine)] for j in range(numLine)],
        #                             dtype= np.int32 ).reshape(numLine*numLine, 3)
        
    def __init_coordinate_xy__(self, x_0, y_0, z_0, numLine):
        return np.array([
                [(x_0 + i, y_0, z_0) for i in range(numLine)],
                [(x_0 + i, -y_0, z_0) for i in range(numLine)],
                [(x_0, y_0 + i, z_0) for i in range(numLine)],
                [(-x_0, y_0 + i, z_0) for i in range(numLine)]
                ]).reshape(numLine*4, 3)
        # return np.array( [[(x_0 + j, y_0 + i, z_0) for i in range(numLine)] for j in range(numLine)],
        #                             dtype= np.int32 ).reshape(numLine*numLine, 3)
    
    def rotate(self, theta= 0, axis= "x"):
        
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
        
        rotate = __rotate__(theta, axis)
        
        # self.coordinate_xy = np.dot(self.coordinate_xy, rotate)
        # self.coordinate_yz = np.dot(self.coordinate_yz, rotate)
        # self.coordinate_xz = np.dot(self.coordinate_xz, rotate)
        for idx, square in enumerate(self.squares):
            self.squares[idx] = np.dot(square, rotate)
        self.normal_vector = np.dot(self.normal_vector, rotate)
    
    def reset(self):
        self.coordinate_xy = self.__init_coordinate_xy__(-10, -10, 10, 20)
        self.coordinate_yz = self.__init_coordinate_yz__(-10, -10, 10, 20)
        self.coordinate_xz = self.__init_coordinate_xz__(-10, -10, 10, 20)
        pass
    
    def writeBuf(self, x_bias= 30, y_bias= 40):
        row = self.size[0]
        col = self.size[1]
        dot_product = np.dot(self.normal_vector, self.visual_vector)
        self.dot_product = dot_product
        # print(dot_product)
        def square2buf(cube: Cube, square, light, x_bias, y_bias, index):
            for c in square:
                cube.buf[int(c[0]+x_bias)%row, int(2*c[1]+y_bias)%col] = light
            c_center = ( square[0] + square[-1] ) // 2
            cube.buf[int(c_center[0]+x_bias)%row, int(2*c_center[1]+y_bias)%col] = index + 2
            
        for i, square in enumerate(self.squares):
            # 依照投影向量与各个面法向量的点积, 决定当前应该显示哪个面
            if dot_product[i] > 0:
                square2buf(self, square, 1, x_bias, y_bias, i)  
    
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
        # print(self.dot_product)

def handler_w(event):
    cube.rotate(theta= np.pi/16, axis= "y")
    cube.writeBuf(x_bias, y_bias)
    cube.show()
    pass

def handler_s(event):
    cube.rotate(theta= -np.pi/16, axis= "y")
    cube.writeBuf(x_bias, y_bias)
    cube.show()
    pass

def handler_d(event):
    cube.rotate(theta= np.pi/16, axis= "x")
    cube.writeBuf(x_bias, y_bias)
    cube.show()
    pass

def handler_a(event):
    cube.rotate(theta= -np.pi/16, axis= "x")
    cube.writeBuf(x_bias, y_bias)
    cube.show()
    pass

def handler_reset(event):
    cube.reset()
    cube.writeBuf()
    cube.show()
    pass

cube = Cube(x_max, y_max, numLine)
if __name__ == "__main__":
    # cube.__init_coordinate__(20, 10, 10)
    # cube.writeBuf(theta= np.pi/6, axis= "x")d
    keyboard.on_press_key("w", handler_w)
    keyboard.on_press_key("a", handler_a)
    keyboard.on_press_key("s", handler_s)
    keyboard.on_press_key("d", handler_d)
    keyboard.on_press_key(" ", handler_reset)
    keyboard.wait("esc")
    # cube.rotate(theta= np.pi/6, axis= "y")
    # cube.rotate(theta= np.pi/6, axis= "y")
    # cube.rotate(theta= np.pi/4, axis= "x")
    
    pass