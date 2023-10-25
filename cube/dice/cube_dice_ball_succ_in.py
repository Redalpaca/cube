import numpy as np
import keyboard
import sys    

numLine = 50
x_max = numLine*2
y_max = numLine*4
x_bias = numLine
y_bias = numLine*2

theta = np.pi / 16

class Surface(object):
    def __init__(self):
        self.points = []
        self.norm_vec = []
    
    def init_points(self, points):
        self.points = points
    pass

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
        self.normal_vector = np.array([[1,0,0],
                                       [-1,0,0],
                                       [0,1,0],
                                       [0,-1,0],
                                       [0,0,1],
                                       [0,0,-1]])
        self.visual_vector = np.array([0,0,1])
        self.illuminant_vector = np.array([0.5773502,0.5773502,0.5773502]) 
        refer = numLine // 2
        
        self.normal_vector_ball = [[] for i in range(6)]
        
        # 6
        self.ball_yz = [self.__init_coordinate_yz_ball__(refer, numLine//6, i*numLine//4, numLine= numLine//5, direct= -1) for i in range(-1,2)]
        self.ball_yz.extend([self.__init_coordinate_yz_ball__(refer, -numLine//6, i*numLine//4, numLine= numLine//5, direct= -1) for i in range(-1,2)])
        # 1
        self.ball_yz_1 = [self.__init_coordinate_yz_ball__(-refer, 0, 0, numLine= numLine//5, direct= 1)]
        
        # 3
        self.ball_xz = [self.__init_coordinate_xz_ball__(i*numLine//4, refer, 0, numLine= numLine//5, direct= -1) for i in range(-1,2)]
        # 4
        self.ball_xz_1 = [self.__init_coordinate_xz_ball__(i*numLine//6, -refer, -numLine//6 , numLine= numLine//5, direct= 1) for i in [-1,1]]
        self.ball_xz_1.extend(self.__init_coordinate_xz_ball__(i*numLine//6, -refer, numLine//6, numLine= numLine//5, direct= 1) for i in [-1,1])
        
        # 2
        self.ball_xy = [self.__init_coordinate_xy_ball__(i*numLine//6, 0, refer, numLine= numLine//5, direct= -1) for i in [-1,1]]
        # 5
        self.ball_xy_1 = [self.__init_coordinate_xy_ball__(i*numLine//4, -numLine//4, -refer, numLine= numLine//5, direct= 1) for i in [-1,1]]
        self.ball_xy_1.extend(self.__init_coordinate_xy_ball__(i*numLine//4, numLine//4, -refer, numLine= numLine//5, direct= 1) for i in [-1,1])
        self.ball_xy_1.append(self.__init_coordinate_xy_ball__(0, 0, -refer, numLine= numLine//5, direct= 1))
        
        self.coordinate_yz = self.__init_coordinate_yz__(-refer, -refer, refer, numLine= numLine)
        self.coordinate_xz = self.__init_coordinate_xz__(-refer, -refer, refer, numLine= numLine)
        self.coordinate_xy = self.__init_coordinate_xy__(-refer, -refer, refer, numLine= numLine)
        self.coordinate_yz_1 = self.__init_coordinate_yz__(refer, -refer, refer, numLine= numLine)
        self.coordinate_xz_1 = self.__init_coordinate_xz__(-refer, refer, refer, numLine= numLine)
        self.coordinate_xy_1 = self.__init_coordinate_xy__(-refer, -refer, -refer, numLine= numLine)
        
        # self.squares = [self.coordinate_yz, self.coordinate_xz, self.coordinate_xy,
        #                 self.coordinate_yz_1, self.coordinate_xz_1, self.coordinate_xy_1]
        self.squares = [self.coordinate_yz_1, self.coordinate_yz, self.coordinate_xz_1, self.coordinate_xz, self.coordinate_xy, self.coordinate_xy_1]
        
        self.balls_ = [self.ball_yz, self.ball_yz_1, self.ball_xz, self.ball_xz_1, self.ball_xy, self.ball_xy_1]
        
        # self.squares = [self.coordinate_yz, self.coordinate_xz, self.coordinate_xy]
        self.lightMap = [' ', '.', '1','2','3','6','5','4']
        self.lightMap = " .-:;=+*##%@@@@@@"
        pass
    
    def __init_coordinate_yz_ball__(self, x_0, y_0, z_0, numLine, direct = 1):
        def calc_c(list_:list, x, y, z):
            y = y-y_0
            z = z-z_0
            if y**2 + z**2 <= numLine*numLine / 4:
                x_ = (x + direct*(numLine**2 / 4 - y**2 - z**2)**0.5/2)
                norm_vec.append((x_-x_0,y,z))
                list_.append((x_, y+y_0, z+z_0))
        ball = []
        norm_vec = []
        for i in range(-numLine//2, numLine//2):
            for j in range(-numLine//2, numLine//2):
                calc_c(ball, x_0, y_0+i, z_0+j)
        if direct == -1:
            self.normal_vector_ball[0].append( np.array(norm_vec) )
        else:
            self.normal_vector_ball[1].append( np.array(norm_vec) )
        return np.array(ball)
    
    def __init_coordinate_xz_ball__(self, x_0, y_0, z_0, numLine, direct = 1):
        def calc_c(list_:list, x, y, z):
            x = x-x_0
            z = z-z_0
            if x**2 + z**2 <= numLine*numLine / 4:
                y_ = (y + direct*(numLine**2 / 4 - x**2 - z**2)**0.5/2)
                norm_vec.append((x,y_-y_0,z))
                list_.append((x+x_0, y_, z+z_0))
        ball = []
        norm_vec = []
        for i in range(-numLine//2, numLine//2):
            for j in range(-numLine//2, numLine//2):
                calc_c(ball, x_0+i, y_0, z_0+j)
        if direct == -1:
            self.normal_vector_ball[2].append( np.array(norm_vec) )
        else:
            self.normal_vector_ball[3].append( np.array(norm_vec) )
        return np.array(ball)
    
    def __init_coordinate_xy_ball__(self, x_0, y_0, z_0, numLine, direct = 1):
        def calc_c(list_:list, x, y, z):
            x = x-x_0
            y = y-y_0
            if x**2 + y**2 <= numLine*numLine / 4:
                z_ = (z+ direct*(numLine**2 / 4 - x**2 - y**2)**0.5/2)
                norm_vec.append((x,y,z_-z_0))
                list_.append((x+x_0, y+y_0 , z_))
        ball = []
        norm_vec = []
        for i in range(-numLine//2, numLine//2):
            for j in range(-numLine//2, numLine//2):
                calc_c(ball, x_0+i, y_0+j, z_0)
        if direct == -1:
            self.normal_vector_ball[4].append( np.array(norm_vec) )
        else:
            self.normal_vector_ball[5].append( np.array(norm_vec) )
        return np.array(ball)
    
    def __init_coordinate_yz__(self, x_0, y_0, z_0, numLine):
        return np.array( [[(x_0, y_0 + i, z_0 - j) for i in range(0, numLine, 2)] for j in range(0, numLine, 2)],
                                    dtype= np.int32 ).reshape(numLine*numLine//4, 3)
    
    def __init_coordinate_xz__(self, x_0, y_0, z_0, numLine):
        return np.array( [[(x_0 + i, y_0, z_0 - j) for i in range(0, numLine, 2)] for j in range(0, numLine, 2)],
                                    dtype= np.int32 ).reshape(numLine*numLine//4, 3)
        
    def __init_coordinate_xy__(self, x_0, y_0, z_0, numLine):
        return np.array( [[(x_0 + j, y_0 + i, z_0) for i in range(0, numLine, 2)] for j in range(0, numLine, 2)],
                                    dtype= np.int32 ).reshape(numLine*numLine//4, 3)
    
    # def __init_coordinate_yz__(self, x_0, y_0, z_0, numLine):
    #     return np.array([
    #             [(x_0, y_0, z_0 - i) for i in range(numLine)],
    #             [(x_0, -y_0, z_0 - i) for i in range(numLine)],
    #             [(x_0, y_0 + i, z_0) for i in range(numLine)],
    #             [(x_0, y_0 + i, -z_0) for i in range(numLine)]
    #             ]).reshape(numLine*4, 3)
    #     # return np.array( [[(x_0, y_0 + i, z_0 - j) for i in range(numLine)] for j in range(numLine)],
    #     #                             dtype= np.int32 ).reshape(numLine*numLine, 3)
    
    # def __init_coordinate_xz__(self, x_0, y_0, z_0, numLine):
    #     return np.array([
    #             [(x_0 + i, y_0, z_0) for i in range(numLine)],
    #             [(x_0 + i, y_0, -z_0) for i in range(numLine)],
    #             [(x_0, y_0, z_0 - i) for i in range(numLine)],
    #             [(-x_0, y_0, z_0 - i) for i in range(numLine)]
    #             ]).reshape(numLine*4, 3)
    #     # return np.array( [[(x_0 + i, y_0, z_0 - j) for i in range(numLine)] for j in range(numLine)],
    #     #                             dtype= np.int32 ).reshape(numLine*numLine, 3)
        
    # def __init_coordinate_xy__(self, x_0, y_0, z_0, numLine):
    #     return np.array([
    #             [(x_0 + i, y_0, z_0) for i in range(numLine)],
    #             [(x_0 + i, -y_0, z_0) for i in range(numLine)],
    #             [(x_0, y_0 + i, z_0) for i in range(numLine)],
    #             [(-x_0, y_0 + i, z_0) for i in range(numLine)]
    #             ]).reshape(numLine*4, 3)
    #     # return np.array( [[(x_0 + j, y_0 + i, z_0) for i in range(numLine)] for j in range(numLine)],
    #     #                             dtype= np.int32 ).reshape(numLine*numLine, 3)
    
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
        
        for idx, square in enumerate(self.squares):
            self.squares[idx] = np.dot(square, rotate)
        
        for i, balls in enumerate(self.balls_):
            for j, ball in enumerate(balls):
                self.normal_vector_ball[i][j] = np.dot(self.normal_vector_ball[i][j], rotate)
                balls[j] = np.dot(ball, rotate)
        
        self.normal_vector = np.dot(self.normal_vector, rotate)
    
    def rotate_light(self, theta= 0, axis= "x"):
        rotate = Cube.__rotate__(theta, axis)
        self.illuminant_vector = np.dot(self.illuminant_vector, rotate)
        pass
    
    def reset(self):
        self.coordinate_xy = self.__init_coordinate_xy__(-10, -10, 10, 20)
        self.coordinate_yz = self.__init_coordinate_yz__(-10, -10, 10, 20)
        self.coordinate_xz = self.__init_coordinate_xz__(-10, -10, 10, 20)
        pass
    
    def writeBuf(self, x_bias= 30, y_bias= 40):
        row = self.size[0]
        col = self.size[1]
        
        light_product = np.dot(self.normal_vector, self.illuminant_vector) * 3 + 3
        
        dot_product = np.dot(self.normal_vector, self.visual_vector)
        self.dot_product = dot_product
        # print(dot_product)
        def square2buf(cube: Cube, square, light, x_bias, y_bias, index):
            for c in square:
                cube.buf[int(c[0]+x_bias)%row, int(2*c[1]+y_bias)%col] = light
            # c_center = ( square[0] + square[-1] ) // 2
            # cube.buf[int(c_center[0]+x_bias)%row, int(2*c_center[1]+y_bias)%col] = index + 2
            
        for i, square in enumerate(self.squares):
            # 依照投影向量与各个面法向量的点积, 决定当前应该显示哪个面
            if dot_product[i] > 0:
                square2buf(self, square, light_product[i], x_bias, y_bias, i)  
        
        for i, balls in enumerate(self.balls_):
            if dot_product[i] > 0.2:
                for j, ball in enumerate(balls):
                    lights = np.dot(-self.normal_vector_ball[i][j], self.illuminant_vector) / (self.numLine//5) * 6 + 7
                    for k, c in enumerate(ball):
                        if self.normal_vector_ball[i][j][k][2] < 0:
                            light = lights[i]#np.dot(self.normal_vector_ball[i][j][k], self.illuminant_vector) / (self.numLine//5) * 5 + 6
                            cube.buf[int(c[0]+x_bias)%row, int(2*c[1]+y_bias)%col] = light
                # square2buf(self, ball, 1, x_bias, y_bias, i)
    
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