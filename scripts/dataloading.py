from phonotune.alexandria.pair_constructor import PairConstructor
from phonotune.alexandria.phonon_data import PhononData

mp_id = "mp-556756"
data = PhononData.load_phonon_data(mp_id)
pc = PairConstructor(data)
pairs = pc.construct_all_pairs()

print(len(pairs))
