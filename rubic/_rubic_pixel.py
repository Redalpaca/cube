import numpy as np
import keyboard
import sys  
import os  

""" 使用方法
123456  选择层数
wasd    旋转视角
上下左右 魔方自旋
x       更新魔方面(已有自动更新)
"""

numLine = 80
x_max = numLine*1
y_max = numLine*2
x_bias = numLine//2
y_bias = numLine

layer_cur = 1
theta = np.pi/8
theta_cur = 0

def rotate_n(n1, n2, n3, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    c = 1-np.cos(theta)
    return np.array([[a+c*n1**2, c*n1*n2-b*n3, c*n1*n3+b*n2],
                     [c*n1*n2+b*n3, a+c*n2**2, c*n2*n3-b*n1],
                     [c*n1*n3-b*n2, c*n2*n3+b*n1, a+c*n3**2]])

class Cube(object):
    def __init__(self, row, col, numLine= 20): 
        self.size = (row, col)
        self.numLine = numLine
        
        self.buf = np.zeros((row, col))
        self.current_rotate = np.identity(3, np.float32)
        self.accumulate_rotate = np.identity(3, np.float32)
        
        refer = numLine // 2
        _len_ = numLine // 3
        
        self.yz_0 = [] 
        self.yz_1 = []
        self.xz_0 = []
        self.xz_1 = []
        self.xy_0 = []
        self.xy_1 = []
        
        gap = 4
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                self.yz_0.append(self.__init_yz__(refer, _len_*i, _len_*j, _len_ - gap))
                self.yz_1.append(self.__init_yz__(-refer, _len_*i, _len_*j, _len_ - gap))
                self.xz_0.append(self.__init_xz__(_len_*i, refer ,_len_*j, _len_ - gap))
                self.xz_1.append(self.__init_xz__(_len_*i, -refer, _len_*j, _len_ - gap))
                self.xy_0.append(self.__init_xy__(_len_*i, _len_*j, refer, _len_ - gap))
                self.xy_1.append(self.__init_xy__(_len_*i, _len_*j, -refer, _len_ - gap))
        self.base_vector = np.array([[1,0,0],
                                     [0,1,0],
                                     [0,0,1]])
        self.normal_vector = np.array([
                                       [[1,0,0] for i in range(9)],
                                       [[-1,0,0]for i in range(9)],
                                       [[0,1,0] for i in range(9)],
                                       [[0,-1,0]for i in range(9)],
                                       [[0,0,1] for i in range(9)],
                                       [[0,0,-1]for i in range(9)]
                                       ], dtype= np.float32)
        self.visual_vector = np.array([0,0,1])
        self.squares = [self.yz_0, self.yz_1, self.xz_0, self.xz_1, self.xy_0, self.xy_1]
        self.lightMap = [" ", "\033[32m█\033[0m","\033[34m█\033[0m","\033[31m█\033[0m","\033[38;5;214m█\033[0m","\033[37m█\033[0m","\033[33m█\033[0m"]
        self.sign_map = np.zeros((6,9))
        for i in range(6):
            for j in range(9):
                self.sign_map[i][j] = i+1
    
    def __init_yz__(self, x_0, y_0, z_0, numLine):
        return np.array( [[(x_0, y_0 + i, z_0 + j) for i in range(-numLine//2 ,numLine//2)] for j in range(-numLine//2, numLine//2)],
                            dtype= np.float64 ).reshape(numLine*numLine, 3)
    def __init_xz__(self, x_0, y_0, z_0, numLine):
        return np.array( [[(x_0 + i, y_0, z_0 + j) for i in range(-numLine//2 ,numLine//2)] for j in range(-numLine//2, numLine//2)],
                            dtype= np.float64 ).reshape(numLine*numLine, 3)
    def __init_xy__(self, x_0, y_0, z_0, numLine):
        return np.array( [[(x_0 + i, y_0 + j, z_0) for i in range(-numLine//2 ,numLine//2)] for j in range(-numLine//2, numLine//2)],
                            dtype= np.float64 ).reshape(numLine*numLine, 3)

    def __rotate__(theta= 0, axis= "x"):
        def rotate_x(theta):
            return np.array([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)] ], dtype= np.float32)
        def rotate_y(theta):
            return np.array([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)] ], dtype= np.float32)
        def rotate_z(theta):
            return np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1] ], dtype= np.float32)
        AxisMap = {"x":rotate_x, "y":rotate_y, "z":rotate_z}
        return AxisMap[axis](theta)
    
    def rotate(self, theta= 0, axis= "x"):
        rotate = Cube.__rotate__(theta, axis)
        self.accumulate_rotate = np.dot(self.accumulate_rotate, rotate)        
        for _list_ in self.squares:
            for idx, square in enumerate(_list_):
                _list_[idx] = np.dot(square, rotate)
        for idx, norm_vec in enumerate(self.normal_vector):
            self.normal_vector[idx] = np.dot(norm_vec, rotate)
        self.base_vector = np.dot(self.base_vector, rotate)
    
    def rotate_rubic(self, theta= 0, layer= 1):
        def rotate_by_layer_row(layer, _range_: tuple, rotate):
            begin = (layer-1)*3
            for i in _range_:
                for j in range(begin, begin+3):
                    self.squares[i][j]       = np.dot(self.squares[i][j], rotate)
                    self.normal_vector[i][j] = np.dot(self.normal_vector[i][j], rotate)
                    
        def rotate_by_layer_col(layer, _range_: tuple, rotate):
            for i in _range_:
                for j in range(3):
                    j_ = (layer-1) + 3*j
                    self.squares[i][j_]       = np.dot(self.squares[i][j_], rotate)
                    self.normal_vector[i][j_] = np.dot(self.normal_vector[i][j_], rotate)

        def rotate_l1_x(cube: Cube, rotate):
            rotate_by_layer_row(1, (2,3,4,5), rotate)
            cube.squares[1] = np.dot(cube.squares[1], rotate)
            cube.normal_vector[1] = np.dot(cube.normal_vector[1], rotate)
        
        def rotate_l2_x(cube: Cube, rotate):
            rotate_by_layer_row(2, (2,3,4,5), rotate)
        
        def rotate_l3_x(cube: Cube, rotate):
            rotate_by_layer_row(3, (2,3,4,5), rotate)
            cube.squares[0] = np.dot(cube.squares[0], rotate)
            cube.normal_vector[0] = np.dot(cube.normal_vector[0], rotate)
        
        def rotate_l1_y(cube: Cube, rotate):
            rotate_by_layer_row(1, (0,1), rotate)
            rotate_by_layer_col(1, (4,5), rotate)
            cube.squares[3] = np.dot(cube.squares[3], rotate)
            cube.normal_vector[3] = np.dot(cube.normal_vector[3], rotate)

        def rotate_l2_y(cube: Cube, rotate):
            rotate_by_layer_row(2, (0,1), rotate)
            rotate_by_layer_col(2, (4,5), rotate)

        def rotate_l3_y(cube: Cube, rotate):
            rotate_by_layer_row(3, (0,1), rotate)
            rotate_by_layer_col(3, (4,5), rotate)
            cube.squares[2] = np.dot(cube.squares[2], rotate)
            cube.normal_vector[2] = np.dot(cube.normal_vector[2], rotate)
            
        if layer in range(1,4):
            axis = "x"
        elif layer in range(4,7):
            axis = "y"
        else:
            return
        rotate = Cube.__rotate__(theta, axis)
        rotate = np.dot(np.dot(self.base_vector.T, rotate), self.base_vector)
        handler_map = [rotate_l1_x, rotate_l2_x, rotate_l3_x, rotate_l1_y, rotate_l2_y, rotate_l3_y]
        handler_map[layer-1](self, rotate)
    
    def update(self, layer):
        def shift_x(idx, direct = 1):
            if direct == 1:
                g1 = (1,3,7,5)
                g2 = (2,0,6,8)
            elif direct == 2:
                g1 = (1,5,7,3)
                g2 = (2,8,6,0)
            else:
                return
            s = self.squares[idx]
            m = self.sign_map[idx]
            temp_s_1 = s[g1[0]].copy()
            temp_s_2 = s[g2[0]].copy()
            temp_m_1 = m[g1[0]]
            temp_m_2 = m[g2[0]]
            for i in range(3):
                s[g1[i]] = s[g1[i+1]].copy()
                s[g2[i]] = s[g2[i+1]].copy()
                m[g1[i]] = m[g1[i+1]]
                m[g2[i]] = m[g2[i+1]]
            s[g1[-1]] = temp_s_1
            s[g2[-1]] = temp_s_2
            m[g1[-1]] = temp_m_1
            m[g2[-1]] = temp_m_2
            
        def selfrotate(rs, ri):
            # ATTENTION: COPY !!!! 
            s = self.squares
            v = self.normal_vector
            m = self.sign_map
            temp_s = [s[rs[0]][ri[0][i]].copy() for i in range(3)]
            temp_v = [v[rs[0]][ri[0][i]].copy() for i in range(3)]
            temp_m = [m[rs[0]][ri[0][i]] for i in range(3)]
            for i in range(3):
                for j in range(3):
                    s[rs[i]][ri[i][j]] = s[rs[i+1]][ri[i+1][j]].copy()
                    v[rs[i]][ri[i][j]] = v[rs[i+1]][ri[i+1][j]].copy()
                    m[rs[i]][ri[i][j]] = m[rs[i+1]][ri[i+1][j]]
            for i in range(3):
                s[rs[-1]][ri[-1][i]] = temp_s[i]
                v[rs[-1]][ri[-1][i]] = temp_v[i]
                m[rs[-1]][ri[-1][i]] = temp_m[i]
        
        if layer in range(1,4):
            rs = (2,4,3,5)
            ri = np.array([[2,1,0],[0,1,2],[0,1,2],[2,1,0]]) 
            ri = ri + 3*(layer-1)
            selfrotate(rs, ri)
            if layer == 1:
                shift_x(1, direct= 1)
            elif layer == 3: 
                shift_x(0, direct= 1)
        elif layer in range(4,7):
            rs = (1,4,0,5)
            if layer == 4:
                ri = [[0,1,2],[0,3,6],[2,1,0],[6,3,0]]
                selfrotate(rs, ri)
                shift_x(3, direct= 2)
            elif layer == 5:
                ri = [[3,4,5],[1,4,7],[5,4,3],[7,4,1]]
                selfrotate(rs, ri)
            elif layer == 6:
                ri = [[6,7,8],[2,5,8],[8,7,6],[8,5,2]]
                selfrotate(rs, ri)
                shift_x(2, direct= 2)
        print("layer=", layer)
        
    def reset(self):
        # TODO
        pass
    
    def writeBuf(self, x_bias= 30, y_bias= 40):
        def square2buf(cube: Cube, square, light, x_bias, y_bias):
            for i, c in enumerate(square):
                x = int(c[0]/2+x_bias)%row
                y = int(c[1]+y_bias)%col
                cube.buf[x, y] = light
            
            # for i, c in enumerate(square):
            #     x = int(c[0]+x_bias)%row
            #     y = int(2*c[1]+y_bias)%col
            #     cube.buf[x, y] = light    
            
            # for i in range(len(square)-1):
            #     c = square[i]
            #     c_ = square[i+1]
            #     x = int((c[0]+c_[0])/2+x_bias)%row
            #     y = int((c[1]+c_[1])+y_bias)%col
            #     cube.buf[x, y] = light
            #     pass
        
        row = self.size[0]
        col = self.size[1]
        dot_product = [np.dot(self.normal_vector[i], self.visual_vector) for i in range(6)]
                    
        for i, _list_ in enumerate(self.squares):
            for j, square in enumerate(_list_):
                if dot_product[i][j] > 0:
                    square2buf(self, square, self.sign_map[i][j], x_bias, y_bias)
    
    def show(self):
        def fresh():
            os.system("cls")
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.flush()
        fresh()
        for row in self.buf:
            for column in row:
                print(self.lightMap[int(column)], end='');
            print('')
        self.buf = np.zeros_like(self.buf)

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

def handler_up(event):
    cube.rotate_rubic(0, np.pi/8, axis= "y")
    cube.writeBuf(x_bias, y_bias)
    cube.show()
    pass

def handler_rubic(event):
    global layer_cur
    global theta
    global theta_cur
    if event.name in "123456":
        layer_cur = int(event.name)
    if event.name in ["up", "down", "left", "right"]:
        if event.name in ["up", "right"]:
            theta_ = theta
            theta_cur += theta
        else:
            theta_ = -theta
            theta_cur -= theta
        cube.rotate_rubic(theta_, layer_cur)
        cube.writeBuf(x_bias, y_bias)
        cube.show()
        if theta_cur == np.pi/2:
            theta_cur = 0
            cube.update(layer_cur)
        if theta_cur == -np.pi/2:
            theta_cur = 0
            cube.update(layer_cur)
            cube.update(layer_cur)
            cube.update(layer_cur)
    if event.name == "x":
        cube.update(layer_cur)
        
cube = Cube(x_max, y_max, numLine)
if __name__ == "__main__":
    keyboard.on_press_key("w", handler_w)
    keyboard.on_press_key("a", handler_a)
    keyboard.on_press_key("s", handler_s)
    keyboard.on_press_key("d", handler_d)
    keyboard.on_press(handler_rubic)
    keyboard.on_press_key(" ", handler_reset)
    keyboard.wait("esc")
    