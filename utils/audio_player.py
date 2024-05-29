from audioplayer import AudioPlayer

def audioPlay(jutsu):
    global effect
    effect = AudioPlayer(f"audio/{jutsu}.mp3")
    effect.play(block=False)
