import sys
import copy
import time
from random import shuffle, seed as random_seed, randrange

class Sudoku(object):
    def __init__(self, width, height, puzzle=None, difficulty=-1, seed=randrange(sys.maxsize)):
        self.puzzle = puzzle
        self.neighbours = {}
        self.height = height
        self.width = width
        self.size = self.height * self.width    # 9
        self.difficulty = difficulty
        self.flag = True
        if puzzle:
            blank_count = 0
            for row in self.puzzle:
                for i in range(len(row)):
                    if type(row[i]) is not int or not 1 <= row[i] <= self.size:
                        row[i] = 0
                        blank_count += 1
            if difficulty == -1:
                self.difficulty = blank_count / self.size / self.size
        else:
            positions = list(range(self.size))
            random_seed(seed)
            shuffle(positions)
            self.puzzle = [[(i + 1) if i == positions[j] else 0 for i in range(self.size)] for j in range(self.size)]   #9*9每行只有一个已知数
            self.difficulty = 0
        self.ans = copy.deepcopy(self.puzzle)

    def solve(self):
        start = time.clock()
        self.flag = self.initialise()
        if not self.flag:
            self.ans = False
            self.difficulty = -1
            print('Invalid puzzle. Please solve the puzzle (puzzle.solve()), or set a difficulty (puzzle.difficulty())')
            return self.ans

        self.flag = self.backtrack(self.puzzle)
        if not self.flag:
            self.ans = False
            self.difficulty = -2
            print('Puzzle has no solution')
            return self.ans

        self.convert_to_output()
        end = time.clock()
        print("[INFO] solving sudoku puzzle cost time {} s: ".format(end - start))
        return self.ans

    def show(self):
        if not self.ans:
            print("NO SOLUTION")
        else:
            print(self.__format_board_ascii())

    def __format_board_ascii(self):
        table = ''
        cell_length = len(str(self.size))
        format_int = '{0:0' + str(cell_length) + 'd}'
        for i, row in enumerate(self.ans):
            if i == 0:
                table += ('+-' + '-' * (cell_length + 1) * self.width) * self.height + '+' + '\n'
            table += (('| ' + '{} ' * self.width) * self.height + '|').format(*[format_int.format(x) if x != 0 else ' ' * cell_length for x in row]) + '\n'
            if i == self.size - 1 or i % self.height == self.height - 1:
                table += ('+-' + '-' * (cell_length + 1) * self.width) * self.height + '+' + '\n'
        return table

    def show_full(self):
        print(self.__str__())

    def __str__(self):
        if self.difficulty == -2:   #无解
            difficulty_str = 'INVALID PUZZLE (GIVEN PUZZLE HAS NO SOLUTION)'
            return '''
---------------------------
{}x{} ({}x{}) SUDOKU PUZZLE
Difficulty: {}
        '''.format(self.size, self.size, self.width, self.height, difficulty_str)
        elif self.difficulty == -1:
            difficulty_str = 'INVALID PUZZLE'
            return '''
---------------------------
{}x{} ({}x{}) SUDOKU PUZZLE
Difficulty: {}
---------------------------
        '''.format(self.size, self.size, self.width, self.height, difficulty_str)
        else:  #解决了
            difficulty_str = 'SOLVED'
        return '''
---------------------------
{}x{} ({}x{}) SUDOKU PUZZLE
Difficulty: {}
---------------------------
{}
        '''.format(self.size, self.size, self.width, self.height, difficulty_str, self.__format_board_ascii())



    def backtrack(self, puzzle):
        if not self.analyse_domains(puzzle):  #pruning
            return False

        h_row, h_col = self.MRV(puzzle)  # Using MRV
        #h_row, h_col = self.MCV(puzzle)  # Using MCV
        if h_row == -1 and h_col == -1:
            self.puzzle = puzzle
            return True

        h_values = puzzle[h_row][h_col].copy()
        for value in h_values:
            tmp_puzzle = copy.deepcopy(puzzle)
            tmp_puzzle[h_row][h_col] = set()
            tmp_puzzle[h_row][h_col].add(value)
            if not self.forward_checking(tmp_puzzle, (h_row, h_col), value):
                continue
            if self.backtrack(tmp_puzzle):
                return True
        return False


    def get_free_neighbours_cells_positions(self, puzzle, position):
        output = list()
        for neighbour in self.neighbours.get(position):
            neighbour_row, neighbour_col = neighbour
            if len(puzzle[neighbour_row][neighbour_col]) != 1:#free
                output.append((neighbour_row, neighbour_col))
        return output

    def MRV(self, puzzle):
        tmp_min = 10
        position = (-1, -1)
        for row in range(9):
            for col in range(9):
                '''
                if (len(puzzle[row][col]) == 0):
                    return (row, col)
                '''
                values = puzzle[row][col]
                if len(values) < tmp_min and len(values) != 1:
                    tmp_min = len(values)
                    position = (row, col)
        return position

    def MCV(self, puzzle):
        position = (-1, -1)
        tmp_max = -1
        for row in range(9):
            for col in range(9):
                '''
                if (len(puzzle[row][col]) == 0):
                    return (row, col)
                '''
                if (len(puzzle[row][col]) != 1):
                    free_connecting_cell_positions = self.get_free_neighbours_cells_positions(puzzle, (row, col))
                    if len(free_connecting_cell_positions) > tmp_max:
                        tmp_max = len(free_connecting_cell_positions)
                        position = (row, col)
        return position


    def get_neighbours_cells_positions(self, puzzle, position):
        row, col = position
        output = list()
        for col_value in range(9):
            if col_value != col:
                output.append((row, col_value))
        for row_value in range(9):
            if row_value != row:
                output.append((row_value, col))
        row_in_box = int(row / 3) * 3
        col_in_box = int(col / 3) * 3
        for box_row_value in range(row_in_box, row_in_box + 3):
            for box_col_value in range(col_in_box, col_in_box + 3):
                if box_row_value != row and box_col_value != col:
                    output.append((box_row_value, box_col_value))
        return output



    def forward_checking(self, puzzle, position, value):
        row, col = position
        puzzle[row][col] = set()
        puzzle[row][col].add(value) #只有一个value
        neighbours_cells = self.neighbours.get(position)#与他有限制的cell
        for neighbours_row, neighbours_col in neighbours_cells:
            if value in puzzle[neighbours_row][neighbours_col]:
                puzzle[neighbours_row][neighbours_col].remove(value)#去掉相同的值
                #如果我们删除了一个值，并且只剩下一个合法的域值，我们实际上是将剩余的那个值分配给该位置。因此，我们必须进行下一轮的领域缩减和前向链
                if len(puzzle[neighbours_row][neighbours_col]) == 1:
                    for remain_value in puzzle[neighbours_row][neighbours_col]:
                        if not self.forward_checking(puzzle, (neighbours_row, neighbours_col), remain_value):
                            return False
            if len(puzzle[neighbours_row][neighbours_col]) == 0:#没有值能选了
                return False
        return True

    def initialise(self):
        new_puzzle = [[set() for i in range(self.size)] for i in range(self.size)]
        for row in range(self.size):    #每个cell初始化有9个备选数
            for col in range(self.size):
                new_puzzle[row][col] = set()
                for i in range(1, self.size+1):
                    new_puzzle[row][col].add(i)
                neighbours = self.get_neighbours_cells_positions(self.puzzle, (row, col))   #里面全是与他有限制的cell
                self.neighbours[(row, col)] = neighbours

        for row in range(self.size):
            for col in range(self.size):
                if self.puzzle[row][col] != 0:
                    if not self.forward_checking(new_puzzle, (row, col), self.puzzle[row][col]):
                        print("Initialize puzzle failed")
                        return False
        self.puzzle = new_puzzle
        return True

    def convert_to_output(self):
        for row in range(9):
            for col in range(9):
                self.ans[row][col] = next(iter(self.puzzle[row][col]))

    def analyse_domains(self, puzzle):
        for row in range(9):
            frequency = {}
            location = {}
            for col in range(9):
                '''
                if len(puzzle[row][col]) == 1:
                    assigned_value = next(iter(puzzle[row][col]))
                    frequency[assigned_value] = -sys.maxsize
                '''
                for value in puzzle[row][col]:#frequency用来存储value的次数
                    if value not in frequency:
                        frequency[value] = 1
                    else:
                        frequency[value] = frequency[value] + 1
                    location[value] = col  # 最后一次的col
            for tmp_value in range(1, 10):
                if tmp_value in frequency and tmp_value in location:
                    if frequency[tmp_value] == 1:
                        if not self.forward_checking(puzzle, (row, location[tmp_value]), tmp_value):
                            return False

        for col in range(9):
            frequency = {}
            location = {}
            for row in range(9):
                '''
                if len(puzzle[row][col]) == 1:
                    assigned_value = next(iter(puzzle[row][col]))
                    frequency[assigned_value] = -sys.maxsize
                '''
                for value in puzzle[row][col]:
                    if value not in frequency:
                        frequency[value] = 1
                    else:
                        frequency[value] = frequency[value] + 1
                    location[value] = row
            for tmp_value in range(1, 10):
                if tmp_value in frequency and tmp_value in location:
                    if frequency[tmp_value] == 1:
                        if not self.forward_checking(puzzle, (location[tmp_value], col), tmp_value):
                            return False

        small_boxes = []
        large_boxes = []
        boxes_list = []
        for i in range(3):
            for j in range(3):
                small_boxes.append((i, j))
                large_boxes.append((i * 3, j * 3))
        for large_box in large_boxes:
            boxes = []
            for small_box in small_boxes:
                boxes.append((small_box[0] + large_box[0], small_box[1] + large_box[1]))
            boxes_list.append(boxes)

        for boxes in boxes_list:
            frequency = {}
            location = {}
            for cell in boxes:
                row = cell[0]
                col = cell[1]
                '''
                if len(puzzle[row][col]) == 1:
                    assigned_value = next(iter(puzzle[row][col]))
                    frequency[assigned_value] = -sys.maxsize
                '''
                for value in puzzle[row][col]:
                    if value not in frequency:
                        frequency[value] = 1
                    else:
                        frequency[value] = frequency[value] + 1
                    location[value] = cell
            for tmp_value in range(1, 10):
                if tmp_value in frequency and tmp_value in location:
                    if frequency[tmp_value] == 1:
                        if not self.forward_checking(puzzle, location[tmp_value], tmp_value):
                            return False

        return True
