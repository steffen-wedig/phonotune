from phonotune.evaluation.phonon_benchmark import Visualizer

models = "mace-omat-0-medium"

vis = Visualizer(models, N_materials=20)
vis.print_td_maes()
violin_fig = vis.make_violin_plots()

violin_fig.savefig("benchmark_violin.png")
