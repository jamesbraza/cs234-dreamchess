import chess
import azg_chess.players as players



#os.environ["CUDA_VISIBLE_DEVICES"]=""



b = chess.Board()
#b.push(chess.Move.from_uci("e2e4"))

p1 = players.StockfishChessPlayer(engine_path='/usr/games/stockfish')
p2 = players.GeochriPlayer(nn_path='model_data/iter4_100.tar')

print(b)


while not (b.is_game_over()):
    m = p1.choose_move(b)
    print(m.uci())
    b.push(m)
    print(b)
    print("-----")

    if b.is_game_over():
        break
    m = p2.choose_move(b)
    print(m.uci())
    b.push(m)
    print(b)
    print("-----")






