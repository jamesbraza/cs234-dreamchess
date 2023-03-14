chess_api_board = chess.Board()

# print(board.legal_moves)


filename = '../data/chess-games/sample.txt'
f = open(filename, "r")
Lines = f.readlines()

for line in Lines[:6]:
    if line[0] != '#':
        line = line.strip()
        t = line.split("###")
        split_head = t[0].split(" ")
        result = split_head[2]
        body = t[1]
        if body == "":
            continue
        pgn = io.StringIO(body)
        current_game = chess.pgn.read_game(pgn)


dataset_p = []

value = 0
if result == "1-0":
    value = 1
if result == "0-1":
    value = -1

geochri_board = c_board()
chess_api_board = current_game.board()


for move in current_game.mainline_moves():
    m = move.uci()
    print(m)

    if len(m) == 5:
        promoted_to = m[4].upper()
    else:
        promoted_to = None

    initial_pos = (7 - (int(m[1:2]) - 1), string.ascii_lowercase.index(m[0:1]))
    final_pos = (7 - (int(m[3:4]) - 1), string.ascii_lowercase.index(m[2:3]))

    policy_idx = copy.deepcopy(ed.encode_action(geochri_board, initial_pos, final_pos, promoted_to))
    geochri_board.move_piece(initial_pos, final_pos, promoted_to)
    chess_api_board.push(move)

    if initial_pos == (0, 4) and final_pos == (0, 6):
        geochri_board.move_piece((0, 7), (0, 5))  # move black rook for short castle

    if initial_pos == (0, 4) and final_pos == (0, 2):
        geochri_board.move_piece((0, 0), (0, 3))  # move black rook for long castle

    if initial_pos == (7, 4) and final_pos == (7, 6):
        geochri_board.move_piece((7, 7), (7, 5))  # move white rook for short castle

    if initial_pos == (7, 4) and final_pos == (7, 2):
        geochri_board.move_piece((7, 0), (7, 3))  # move black rook for long castle


    state = copy.deepcopy(ed.encode_board(geochri_board))

    policy = one_hot_policy(policy_idx)

    if make_string_geochri(geochri_board.current_board) != make_string_chess(chess_api_board):
        raise ValueError("ERROR!")

    print(chess_api_board)


    dataset_p.append([state, policy, value])

print(dataset_p)
