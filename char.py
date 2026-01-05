# chars.py
CHARS = "अआइईउऊएऐओऔकखगघचछजझटठडढतथदधनपफबभमयरलवशषसहािीुूेैोौंःँ्"
BLANK = "-"
VOCAB = BLANK + CHARS

char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for c, i in char2idx.items()}
