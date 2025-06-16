# Chinese Restaurant Process Inference

Particularly for the case of single cell classification inference.

__Work In Progress__

To see usage, go to the notebooks and open `usage.ipynb`.

### TODO
- [ ] Fully nested chinese restaurant process (Blei's infinite tree thing)

- [ ] Easier to extract marker genes.
    - At the moment, you can call the following below, and get the concentration parameters for each of the classes, and use that to determine marker genes.

```python
marker_genes = pd.DataFrame({
    k: v.concentration 
    for k, v in crp.classes.items() 
    if len(v.members) > some_threshold
}, index=adata.var_names)
```

- [ ] Some plotting utilities might be nice.
- [ ] Confidence intervals / confidence scores might be useful.
