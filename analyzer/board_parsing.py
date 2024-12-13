""" Parsing of the fen boards
"""
from typing import List


class Board:
    """Class to parse and store a chess board from it's fen notation
    """

    def __init__(self, fen: str):
        self.fen: str = fen
        self.pieces_list = ["p", "r", "n", "b",
                            "q", "k", "P", "R", "N", "B", "Q", "K"]

        def create_grid() -> List[List[bool]]:
            return [[False for _ in range(8)] for _ in range(8)]

        self.boards: List[List[List[bool]]] = []
        for piece in self.pieces_list:
            setattr(self, piece, create_grid())
            self.boards.append(getattr(self, piece))
        self.turn: str = 'w'
        self.halfmove: int = 0
        self.fullmove: int = 0

        self.__parse()

    def __parse_infos(self, turn: str, rock: str, enpassant: str, halfmove: str, fullmove: str):
        if len(turn) != 0 and not turn in "wb":
            raise ValueError(f"The player turn {turn} is invalid")
        self.turn = turn
        # TODO self.rock = (False, False)
        # TODO self.enpassant = False
        try:
            self.halfmove = int(halfmove)
            self.fullmove = int(fullmove)
        except ValueError:
            print("Move value is wrong")
        return

    def __parse_board(self, board: str):
        lines = board.split('/')
        if len(lines) != 8:
            raise ValueError("The board isn't the good size")
        for i, line in enumerate(lines):
            j = 0
            for piece in line:
                if j >= 8:
                    raise ValueError(f"Line {i} is invalid")
                if piece.isdigit():
                    j += int(piece)
                    continue
                if not piece in self.pieces_list:
                    raise ValueError(
                        f"Unknown piece '{piece}' on the board at line {i}")
                getattr(self, piece)[i][j] = True
                j += 1

    def __check_boards(self):
        # TODO check that the number of pieces is ok and that there aren't two pieces on the same line
        pass

    def __parse(self):
        splitted = self.fen.split(' ')
        if len(splitted) != 6:
            raise ValueError("Not enought information")
        self.__parse_board(splitted[0])
        self.__parse_infos(*splitted[1:])
        self.__check_boards()

    def __repr__(self):
        res = ""
        for piece in self.pieces_list:
            res += f"{piece}\n"
            board = getattr(self, piece)
            res += '\n'.join([' '.join([piece if p else '.' for p in line])
                             for line in board])
            res += '\n'
        return res


if __name__ == "__main__":
    boards = [
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
        "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
        "8/8/8/8/8/8/8/k1K5 w - - 0 1",
    ]
    a = Board(boards[0])
    # print(a.boards)
    print(a)
