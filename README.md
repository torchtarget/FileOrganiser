# FileOrganiser

A simple Python tool that learns your existing folder structure and
uses a naive Bayesian classifier to automatically sort new files.

## Usage

```
python -m fileorganiser <training_root> <incoming_dir> <mapping_file> [--apply] [--save-model <path>] [--load-model <path>]
```

- `training_root` - directory that already contains organised folders.
- `incoming_dir` - directory with new files to be sorted.
- `mapping_file` - path to write the predicted file-to-folder mapping.
- `--apply` - actually move the files after writing the mapping. Without this
  flag the tool only writes the mapping.
- `--save-model <path>` - save the trained classifier to the given file.
- `--load-model <path>` - load a previously saved classifier instead of
  learning from `training_root`.

Example:

```
python -m fileorganiser ~/Documents/Organised ~/Downloads ~/mapping.txt --apply
```

This will learn from the folders in `~/Documents/Organised`, predict where
files from `~/Downloads` belong, write the mapping to `~/mapping.txt` and move
the files.
