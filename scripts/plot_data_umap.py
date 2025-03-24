from phonotune.evaluation.low_dim_projection import (
    assemble_full_formula_dataset,
    collect_formulas,
    get_embedding,
    train_reducer,
)
from phonotune.evaluation.plotting_utils import plot_umap_projection

# Plots the UMAP projections of the datasets


embedding_method = "magpie"

formulas_subsets, names = assemble_full_formula_dataset()
formulas = collect_formulas(formulas_subsets)
embeddings = get_embedding(formulas, embedding_method)
reducer = train_reducer(embeddings)

fig = plot_umap_projection(formulas_subsets, names, reducer, embedding_method)

fig.savefig("DimReduction.pdf")
