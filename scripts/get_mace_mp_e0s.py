from mace.calculators import mace_mp

calc = mace_mp("medium")

E0 = calc.models[0].atomic_energies_fn.atomic_energies.cpu().numpy().tolist()
ztable = calc.z_table.zs
print(ztable)
E0_dict = {}
for i, e in zip(ztable, E0, strict=False):
    E0_dict[i] = e


print(E0_dict)

print([i for i in ztable])
