# tests/test_solver.py
import random
import unittest
import time
from game.board import Board
from game.solver import *

class TestSolverMethods:
    
    def __init__(self):
        self.bfs = []
        self.astar = []
        self.weighed_astar = []
        self.dfs = []
        self.iddfs = []
        self.ucs = []
        self.quick_solver = []

    def generate_board(self, size):
        init_pos = (0, 0)
        pattern = []
        colors = ['R', 'B', 'Y']

        for _1 in range(size):
            row = [random.choice(colors) for _2 in range(size)]
            pattern.append(row)

        return Board(size, size, pattern, init_pos)

    def test_board(self, index, size, qs, was, ast, iddfs, ucs, dfs, bfs):
        board = self.generate_board(size)
        goal_board = self.generate_board(size)

        if qs:
            start = time.time()
            path = find_path_quick_solver(board.copy(), goal_board.copy())
            end = time.time()
            total_time = end - start
            print("Quick Solver took "+str(total_time)+" seconds to complete "+str(size)+'x'+str(size))
            self.quick_solver.append((index, size, total_time, len(path)))

        if was:
            start = time.time()
            path = find_path_weighted_astar(board.copy(), goal_board.copy())
            end = time.time()
            total_time = end - start
            print("Weighted A* took "+str(total_time)+" seconds to complete "+str(size)+'x'+str(size))
            self.weighed_astar.append((index, size, total_time, len(path)))

        if ast:
            start = time.time()
            path = find_path_astar(board.copy(), goal_board.copy())
            end = time.time()
            total_time = end - start
            print("A* took "+str(total_time)+" seconds to complete "+str(size)+'x'+str(size))
            self.astar.append((index, size, total_time, len(path)))

        if iddfs:
            start = time.time()
            path = find_path_iddfs(board.copy(), goal_board.copy())
            end = time.time()
            total_time = end - start
            print("IDDFS took "+str(total_time)+" seconds to complete "+str(size)+'x'+str(size))
            self.iddfs.append((index, size, total_time, len(path)))

        if ucs:
            start = time.time()
            path = find_path_ucs(board.copy(), goal_board.copy())
            end = time.time()
            total_time = end - start
            print("UCS took "+str(total_time)+" seconds to complete "+str(size)+'x'+str(size))
            self.ucs.append((index, size, total_time, len(path)))

        if dfs:
            start = time.time()
            path = find_path_dfs(board.copy(), goal_board.copy())
            end = time.time()
            total_time = end - start
            print("DFS took "+str(total_time)+" seconds to complete "+str(size)+'x'+str(size))
            self.dfs.append((index, size, total_time, len(path)))

        if bfs:
            start = time.time()
            path = find_path_bfs(board.copy(), goal_board.copy())
            end = time.time()
            total_time = end - start
            print("BFS took "+str(total_time)+" seconds to complete "+str(size)+'x'+str(size))
            self.bfs.append((index, size, total_time, len(path)))

    def test_solvers(self):
        self.test_board(1, 2, True, True, True, True, True, True, True)
        self.test_board(2, 3, True, True, True, True, True, True, True)
        self.test_board(3, 4, True, True, False, False, False, False, False)
        self.test_board(4, 5, True, True, False, False, False, False, False)
        self.test_board(5, 6, True, True, False, False, False, False, False)
        self.test_board(6, 8, True, False, False, False, False, False, False)
        self.test_board(7, 10, True, False, False, False, False, False, False)

    def get_results(self):
        results = []
        results.append(('bfs', self.bfs))
        results.append(('dfs', self.dfs))
        results.append(('iddfs', self.iddfs))
        results.append(('ucs', self.ucs))
        results.append(('astar', self.astar))
        results.append(('wastar', self.weighed_astar))
        results.append(('qs', self.quick_solver))
        return results

# More tests to be added for different patterns and solver configurations
