class LabelConverter:
    def __init__(self, charset):
        self.charset = charset
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(charset)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(charset)}
        self.blank = 0

    def encode(self, text):
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]

    def decode(self, preds):
        preds = preds.argmax(2)
        preds = preds.permute(1, 0)
        decoded = []
        for pred in preds:
            prev = -1
            word = ''
            for idx in pred:
                if idx != prev and idx != self.blank:
                    word += self.idx_to_char.get(idx.item(), '')
                prev = idx
            decoded.append(word)
        return decoded
