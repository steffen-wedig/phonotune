from itertools import chain

batch = [("A", "B", "C"), ("D", "E", "F"), ("G", "H", "I")]
output = list(chain.from_iterable(zip(*batch, strict=False)))
print(output)
