import sys
from queue import PriorityQueue, LifoQueue
from copy import deepcopy
import time

#
input_filename = sys.argv[1]
dfs_output_filename = sys.argv[2]
astar_output_filename = sys.argv[3]
CHOSEN_H = 0  # chosen heuristic, can be either 0(manhattan) or 1(new h(n))


# start = time.time()

input_puzzle = []

with open(input_filename, 'r') as input_file:
    for line in input_file:
        row_i = [int(x) for x in line.rstrip()]
        input_puzzle.extend(row_i)


def get_manhattan_distance(boardstate) -> int:
    """ returns a manhattan distance from a main block to a goal position."""
    x_pos = boardstate.main_tl_pos[0]
    y_pos = boardstate.main_tl_pos[1]
    return (3 - x_pos) + abs(1 - y_pos)


def heuristic_function(boardstate) -> int:
    """
    Returns a value computed by CHOSEN_H(0:manhattan, 1:advanced)
    :param boardstate: Board
    :return: an integer value that the heuristic returns
    """
    if CHOSEN_H == 0:
        return get_manhattan_distance(boardstate)
    else:  # choice of a new heuristic
        main_tl = boardstate.main_tl_pos
        e_pos0 = boardstate.e_pos0
        e_pos1 = boardstate.e_pos1
        # if the first one is under the bottomleft AND the second one is under the bottom right
        # or vice versa
        if ([main_tl[0] + 2, main_tl[1]] == e_pos0 and [main_tl[0] + 2, main_tl[
                                                                            1] + 1] == e_pos1) or (
                [main_tl[0] + 2, main_tl[1]] == e_pos1 and [main_tl[0] + 2,
                                                            main_tl[
                                                                1] + 1] == e_pos0):
            return get_manhattan_distance(boardstate)
        else:
            return get_manhattan_distance(boardstate) + 1


class Board(object):
    def __init__(self, parent_board=None, blocks=None):
        self.parent_board = parent_board
        self.string_rep = None
        if parent_board is not None:
            self.blocks = blocks
            self.cost = parent_board.cost + 1
            self.main_tl_pos = self.blocks[(2, 2)][0][0]
            self.e_pos0 = self.blocks[(0, 0)][0][0]
            self.e_pos1 = self.blocks[(0, 0)][1][0]

            self.f = self.cost + heuristic_function(self)

        else:
            self.blocks = {}
            self.cost = 0
            self.e_pos0 = None
            self.e_pos1 = None
            self.main_tl_pos = None

    def update_string_rep(self):
        self.string_rep = str(self)

    def is_block_next_to_empty(self, block_tup: tuple, index: int,
                               direction: str) -> list:
        """ Precondition: block_tup is not (0,0,0) and not (0,0,1).
            That is, empty block being next to another empty block is useless.

            Return list containing empty_block_tuple that is next to the block. 0 <= len(list) <= 2
            0 in length means there is no next empty block.
            1 means there is one and that's all they need.
            2 is the block requires two consecutive next empty blocks to move, and the list contains both tuples.
            """
        e_pos1 = self.e_pos0
        e_pos2 = self.e_pos1
        if direction == 'u':  # is empty block is located on the block?
            if block_tup[1] == 2:  # the block needs two empty blocks upwards.
                b_pos1 = self.blocks[block_tup][index][0]
                b_pos2 = self.blocks[block_tup][index][1]
                if e_pos1[0] != e_pos2[0]:  # two empty spots on the same row
                    return []
                if e_pos1[0] != b_pos1[0] - 1:  # same row, upper of b_pos
                    return []
                if e_pos1[1] == b_pos1[1] and e_pos2[1] == b_pos2[1]:
                    return [0, 1]
                elif e_pos1[1] == b_pos2[1] and e_pos2[1] == b_pos1[1]:
                    return [1, 0]
                else:
                    return []
            else:  # the block is single in horizontal direction, so we only need to care if there is one empty spot on the topmost piece of the block
                b_pos1 = self.blocks[block_tup][index][0]
                if (e_pos1[0] == b_pos1[0] - 1) and (e_pos1[1] == b_pos1[1]):
                    return [0]
                elif (e_pos2[0] == b_pos1[0] - 1) and (e_pos2[1] == b_pos1[1]):
                    return [1]
                return []

        if direction == 'd':  # is empty block is located below the block?
            if block_tup[1] == 2:
                if block_tup == (2, 2):
                    b_pos1 = self.blocks[block_tup][index][
                        2]  # index can really only be 0 here
                    b_pos2 = self.blocks[block_tup][index][3]
                else:  # horizontal 1x2 block
                    b_pos1 = self.blocks[block_tup][index][0]
                    b_pos2 = self.blocks[block_tup][index][1]

                if e_pos1[0] != e_pos2[0]:  # two empty spots on the same row
                    return []
                if e_pos1[0] != b_pos1[0] + 1:  # and empt down the block
                    return []

                if e_pos1[1] == b_pos1[1] and e_pos2[1] == b_pos2[1]:
                    return [0, 1]
                elif e_pos1[1] == b_pos2[1] and e_pos2[1] == b_pos1[1]:
                    return [1, 0]

                else:
                    return []

            else:  # the block is single in horizontal direction, so we only need to care if there is one empty spot below the bottommost piece of the block
                if block_tup == (1, 1):
                    b_pos1 = self.blocks[block_tup][index][0]
                else:  # 2x1 block. the bottommost piece is
                    b_pos1 = self.blocks[block_tup][index][1]
                if (e_pos1[0] == b_pos1[0] + 1) and (e_pos1[1] == b_pos1[1]):
                    return [0]
                elif (e_pos2[0] == b_pos1[0] + 1) and (e_pos2[1] == b_pos1[1]):
                    return [1]
                return []

        if direction == 'l':  # is empty block is located left of the block?
            if block_tup[0] == 2:

                if block_tup == (2, 2):
                    b_pos1 = self.blocks[block_tup][index][0]
                    b_pos2 = self.blocks[block_tup][index][2]
                else:  # vertical 2x1
                    b_pos1 = self.blocks[block_tup][index][0]
                    b_pos2 = self.blocks[block_tup][index][1]

                if e_pos1[1] != e_pos2[1]:  # two empty spots on the same column
                    return []
                if e_pos1[1] != b_pos1[1] - 1:  # empt is at the left of piece
                    return []

                if e_pos1[0] == b_pos1[0] and e_pos2[0] == b_pos2[0]:
                    return [0, 1]
                elif e_pos1[0] == b_pos2[0] and e_pos2[0] == b_pos1[0]:
                    return [1, 0]
                else:
                    return []
            else:  # the block is single in vertical direction, so we only need to care if there is one empty spot at the left of the leftmost piece of the block
                b_pos1 = self.blocks[block_tup][index][0]
                if (e_pos1[0] == b_pos1[0]) and (e_pos1[1] == b_pos1[1] - 1):
                    return [0]
                elif (e_pos2[0] == b_pos1[0]) and (e_pos2[1] == b_pos1[1] - 1):
                    return [1]
                return []

        else:  # direction == "right"
            if block_tup[0] == 2:

                if block_tup == (2, 2):
                    b_pos1 = self.blocks[block_tup][index][1]
                    b_pos2 = self.blocks[block_tup][index][3]
                else:  # vertical 2x1
                    b_pos1 = self.blocks[block_tup][index][0]
                    b_pos2 = self.blocks[block_tup][index][1]

                if e_pos1[1] != e_pos2[1]:  # two empty spots on the same column
                    return []
                if e_pos1[1] != b_pos1[1] + 1:  # empt is at the right of piece
                    return []

                if e_pos1[0] == b_pos1[0] and e_pos2[0] == b_pos2[0]:
                    return [0, 1]
                elif e_pos1[0] == b_pos2[0] and e_pos2[0] == b_pos1[0]:
                    return [1, 0]

                else:
                    return []
            else:  # the block is single in vertical direction, so we only need to care if there is one empty spot at the rifgt of the rightmost piece of the block
                if block_tup == (1, 1):
                    b_pos1 = self.blocks[block_tup][index][0]
                else:  # 1x2 block. the rightmost piece is
                    b_pos1 = self.blocks[block_tup][index][1]
                if (e_pos1[0] == b_pos1[0]) and (e_pos1[1] == b_pos1[1] + 1):
                    return [0]
                elif (e_pos2[0] == b_pos1[0]) and (e_pos2[1] == b_pos1[1] + 1):
                    return [1]
                return []

    def has_same_blocks_config(self, other) -> bool:
        """ return a boolean value that tells if self and other boards have
        the same block configuration """
        return self.string_rep == other.string_rep
        # for b1_type in self.blocks:
        #     if other.blocks[b1_type] != self.blocks[b1_type]:
        #         return False
        # return True

    def move_block(self, block_tup, index, direction, emp_index_list):
        """returns the new moved board.
        Precondition: self.is_block_movable(block, direction)
        it is movable, therefore, len(empty_tuple_list) >= 1"""
        new_blocks = deepcopy(self.blocks)
        new = Board(self, new_blocks)

        if direction == 'u':
            for k in range(len(new_blocks[block_tup][index])):
                new_blocks[block_tup][index][k][0] -= 1
                # find equivalent empty block and swap
                if len(emp_index_list) == 2:
                    new.blocks[(0, 0)][k % 2][0][0] += 1
                else:
                    new_blocks[(0, 0)][emp_index_list[0]][0][0] += 1

        elif direction == 'd':
            for k in range(len(new_blocks[block_tup][index])):
                new_blocks[block_tup][index][k][0] += 1
                if len(emp_index_list) == 2:
                    new_blocks[(0, 0)][k % 2][0][0] -= 1
                else:
                    new_blocks[(0, 0)][emp_index_list[0]][0][0] -= 1

        elif direction == 'l':
            for k in range(len(new_blocks[block_tup][index])):
                new_blocks[block_tup][index][k][1] -= 1
                if len(emp_index_list) == 2:
                    new_blocks[(0, 0)][k % 2][0][1] += 1
                else:
                    new_blocks[(0, 0)][emp_index_list[0]][0][1] += 1

        else:  # direction == "right"
            for k in range(len(new_blocks[block_tup][index])):
                new_blocks[block_tup][index][k][1] += 1
                if len(emp_index_list) == 2:
                    new_blocks[(0, 0)][k % 2][0][1] -= 1
                else:
                    new_blocks[(0, 0)][emp_index_list[0]][0][1] -= 1
        new.update_string_rep()
        return new

    def generate_possible_children(self) -> list:
        """A method that combines is_block_next_to_empty and is_movable,
        and move. This method returns a list of possible children(different states)
        that can be obtained by current state.

        Note this method may include a state that has the same configuration
        as its parent's parent.
        """
        possible_children = []
        for blk in self.blocks:
            if blk == (0, 0):
                continue
            for direc in ['u', 'd', 'r', 'l']:
                for blk_index in range(len(self.blocks[blk])):
                    empty_tup_ind = self.is_block_next_to_empty(blk, blk_index,
                                                                direc)
                    if len(empty_tup_ind):
                        n = self.move_block(blk, blk_index, direc,
                                            empty_tup_ind)
                        if self.parent_board is not None and n.has_same_blocks_config(
                                self.parent_board):  # 고쳐
                            continue
                        possible_children.append(n)
        return possible_children

    def is_goal(self) -> bool:
        return self.main_tl_pos == [3, 1]

    def to_nested_list(self):
        flat = [[[], [], [], []], [[], [], [], []], [[], [], [], []],
                [[], [], [], []], [[], [], [], []]]
        # grab_colours12or21 = {2, 3, 4, 5, 6}
        for dim in self.blocks:
            for ind in self.blocks[dim]:
                if dim == (2, 2):
                    for each_pos in ind:
                        y, x = each_pos
                        flat[y][x] = 1
                elif dim == (1, 1):
                    for each_pos in ind:
                        y, x = each_pos
                        flat[y][x] = 4  # fix to 7 if input
                elif dim == (0, 0):
                    for each_pos in ind:
                        y, x = each_pos
                        flat[y][x] = 0
                # else: INPUT
                #     colour = grab_colours12or21.pop()
                #     for each_pos in ind:
                #         y, x = each_pos
                #         flat[y][x] = colour
                elif dim == (2, 1):
                    for each_pos in ind:
                        y, x = each_pos
                        flat[y][x] = 3
                else:
                    for each_pos in ind:
                        y, x = each_pos
                        flat[y][x] = 2
        return flat

    def __str__(self):
        string_rep = ""
        flattened = [item for sublist in self.to_nested_list() for item in
                     sublist]
        string_rep += ''.join(str(e) for e in flattened)
        string_rep = '8' + string_rep
        return string_rep

    def __hash__(self):
        integer = int(self.string_rep)
        return hash(integer)

    def __eq__(self, other) -> bool:
        """ self and other boards are the same board configuration. Note,
        for this we don't care about self.parent_board == other.parent_board and
        their costs. """
        return self.has_same_blocks_config(other)

    def __lt__(self, other):
        return self.f < other.f

    def __gt__(self, other):
        return self.f > other.f


def astar_solver(start: Board):
    """ A solver using an A* Search algorithm.
        The function returns tuple of Board object <goal_board> and
        the <total_cost> which has the total counts in int.
        i.e., Tuple(Board, int)

        To print, you can trace through parent of the <goal_board>
        """
    # 1. Place the starting node into FRONTIER and find its f(n) value.
    frontier = PriorityQueue()

    frontier.put(start)  # f value
    frontier_dic = {start: 0}  # g value
    explored_dic = {}  # string rep of boardstate: g value and parent
    # 2. Then remove the node from FRONTIER, having the smallest f(n) value.
    #    if it is a goal node, then stop and return to success.

    while not frontier.empty():
        curr = frontier.get()
        curr_cost = frontier_dic.pop(curr)

        if curr.is_goal():
            return curr, curr_cost

        explored_dic[curr] = curr_cost  # explored_dic[curr]: g, parent
        # 3. Else, remove the node from FRONTIER, and find all its successors.

        possible_states = curr.generate_possible_children()
        for possible_next_state in possible_states:
            g = curr_cost + 1
            if possible_next_state in explored_dic and g < explored_dic[
                possible_next_state]:
                explored_dic.pop(possible_next_state)
            if possible_next_state in frontier_dic and g < frontier_dic[
                possible_next_state]:
                frontier_dic.pop(possible_next_state)
            if possible_next_state not in frontier_dic and possible_next_state not in explored_dic:
                frontier_dic[possible_next_state] = g
                frontier.put(possible_next_state)

    # 6. Exit (no solution if nothing has been returned)
    return "no possible solution."


def dfs_solver(start: Board):
    """ A solver using an A* Search algorithm.
        The function returns tuple of Board object <goal_board> and
        the <total_cost> which has the total counts in int.
        i.e., Tuple(Board, int)

        To print, you can trace through parent of the <goal_board>
        """
    # 1. Place the starting node into FRONTIER and find its f(n) value.
    frontier = LifoQueue()
    frontier.put(start)  # f value
    explored_set = set()  # string rep of boardstate: g value and parent
    # 2. Then remove the node from FRONTIER, having the smallest f(n) value.
    #    if it is a goal node, then stop and return to success.

    while not frontier.empty():
        curr = frontier.get()

        if curr.is_goal():
            return curr, curr.cost

        explored_set.add(curr)  # explored_dic[curr]: g, parent
        # 3. Else, remove the node from FRONTIER, and find all its successors.

        possible_states = curr.generate_possible_children()
        for possible_next_state in possible_states:
            if possible_next_state not in explored_set:
                frontier.put(possible_next_state)

    # 6. Exit (no solution if nothing has been returned)
    return "no possible solution."


board = Board()

for i in range(len(input_puzzle)):
    if input_puzzle[i] in [2, 3, 4, 5, 6]:
        if i + 4 < len(input_puzzle) and input_puzzle[i] == input_puzzle[
            i + 4]:  # index + 4 포함까지는 봐야됨
            if (2, 1) not in board.blocks:
                board.blocks[(2, 1)] = [
                    [
                        [i // 4, i % 4],
                        [(i + 4) // 4, (i + 4) % 4]
                    ]
                ]
            else:
                board.blocks[(2, 1)].append(
                    [
                        [i // 4, i % 4],
                        [(i + 4) // 4, (i + 4) % 4]
                    ]
                )

        elif i + 1 < len(input_puzzle) and input_puzzle[i] == input_puzzle[
            i + 1]:
            if (1, 2) not in board.blocks:
                board.blocks[(1, 2)] = [
                    [
                        [i // 4, i % 4],
                        [(i + 1) // 4, (i + 1) % 4]
                    ]
                ]
            else:
                board.blocks[(1, 2)].append(
                    [
                        [i // 4, i % 4],
                        [(i + 1) // 4, (i + 1) % 4]
                    ]
                )


        else:  # now it's not find. the only possibility is the piece has been explored.
            continue

    elif input_puzzle[i] == 1:
        if i + 4 < len(input_puzzle) and input_puzzle[i + 1] == 1 and \
                input_puzzle[
                    i + 4] == 1:  # first piece of the block. It should be.
            board.blocks[(2, 2)] = [[[i // 4, i % 4],
                                     [(i + 1) // 4, (i + 1) % 4],
                                     [(i + 4) // 4, (i + 4) % 4],
                                     [(i + 5) // 4, (i + 5) % 4]]]
            board.main_tl_pos = [i // 4, i % 4]

        else:  # explored by the previous piece
            continue

    elif input_puzzle[i] == 7:  # it's a single piece
        if (1, 1) not in board.blocks:
            board.blocks[(1, 1)] = [
                [
                    [i // 4, i % 4]
                ]
            ]
        else:
            board.blocks[(1, 1)].append(
                [
                    [i // 4, i % 4]
                ]
            )
    elif input_puzzle[i] == 0:
        if (0, 0) not in board.blocks:
            board.blocks[(0, 0)] = [
                [
                    [i // 4, i % 4]
                ]
            ]
            board.e_pos0 = [i // 4, i % 4]
        else:
            board.blocks[(0, 0)].append(
                [
                    [i // 4, i % 4]
                ]
            )
            board.e_pos1 = [i // 4, i % 4]
    else:
        continue
board.update_string_rep()
board.f = 0 + heuristic_function(board)
# board.P = board.into_array()  # 디버깅


# DFS ANSWER WRITING

dfs_answer = dfs_solver(board)
dfs_cost = dfs_answer[1]
dfs_paths_initial_to_goal = []
state_looking = dfs_answer[0]
while state_looking != board:
    one_board_str = state_looking.string_rep.strip('8')
    chunks = ""
    for chunk in ['\n' + one_board_str[i:i + 4] for i in range(0, 20, 4)]:
        chunks += chunk
    dfs_paths_initial_to_goal.insert(0, chunks)
    dfs_paths_initial_to_goal.insert(0, '\n')
    state_looking = state_looking.parent_board
one_board_str = state_looking.string_rep.strip('8')
chunks = ""
for chunk in ['\n' + one_board_str[i:i + 4] for i in range(0, 20, 4)]:
    chunks += chunk
dfs_paths_initial_to_goal.insert(0, chunks)

with open(dfs_output_filename, 'w') as dfs_output_file:
    # write the outputs in exact format
    dfs_output_file.writelines(["Cost of the solution: ", str(dfs_answer[1])])
    dfs_output_file.writelines(dfs_paths_initial_to_goal)
    # slicing to get rid of the last new line.

# ASTAR ANSWER WRITING


astar_answer = astar_solver(board)
astar_cost = astar_answer[1]
astar_paths_initial_to_goal = []
state_looking = astar_answer[0]
while state_looking != board:
    one_board_str = state_looking.string_rep.strip('8')
    chunks = ""
    for chunk in ['\n' + one_board_str[i:i + 4] for i in range(0, 20, 4)]:
        chunks += chunk
    astar_paths_initial_to_goal.insert(0, chunks)
    astar_paths_initial_to_goal.insert(0, '\n')
    state_looking = state_looking.parent_board
one_board_str = state_looking.string_rep.strip('8')
chunks = ""
for chunk in ['\n' + one_board_str[i:i + 4] for i in range(0, 20, 4)]:
    chunks += chunk
astar_paths_initial_to_goal.insert(0, chunks)

with open(astar_output_filename, 'w') as astar_output_file:
    # write the outputs in exact format
    astar_output_file.writelines(
        ["Cost of the solution: ", str(astar_answer[1])])
    astar_output_file.writelines(astar_paths_initial_to_goal)
    # slicing to get rid of the last new line.



#
# end = time.time()
# time_took = end - start
# print('Took: {}seconds'.format(time_took))
