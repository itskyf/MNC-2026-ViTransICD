# ViTransICD

## Commands

```shell
# Ontology crawl
uv run -m mnc.ontology.crawl

# DE-02: Snapshot datasets to bronze
uv run -m mnc.datasets.cli vietmed-sum <raw_input_dir>
uv run -m mnc.datasets.cli vihealthqa <raw_input_dir>

# DE-03: Parse bronze to bronze docs
uv run -m mnc.datasets.de3_cli vietmed-sum
uv run -m mnc.datasets.de3_cli vihealthqa
```
