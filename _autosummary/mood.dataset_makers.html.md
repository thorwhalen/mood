# mood.dataset_makers

Tools to make datasets for training and testing models (for semantic attribute etc.)

### Functions

| [`default_save_key`](#mood.dataset_makers.default_save_key)(name[, batch_idx, ...])     | Build the storage key for a dataset batch.                        |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`make_semantic_attributes_dataset`](#mood.dataset_makers.make_semantic_attributes_dataset)(...[, ...]) | Gather data for each semantic attribute and save it to the store. |
| `mk_concatenation_file`(dirpath, \*[, ...])                                                   |                                                                   |
| `parse_line`(line)                                                                            |                                                                   |
| `parsed_lines`(text_contents)                                                                 |                                                                   |
| [`text_lines`](#mood.dataset_makers.text_lines)(text)                             | Yield the non-empty, stripped lines of `text`.                    |

### mood.dataset_makers.default_save_key(name, batch_idx=None, , include_subdir_prefix=True, counter_key_format='_{:02.0f}', extension='.txt')

Build the storage key for a dataset batch.

```pycon
>>> default_save_key("color", include_subdir_prefix=False)
'color.txt'
>>> default_save_key("color", 1, include_subdir_prefix=False)
'color_01.txt'
```

### mood.dataset_makers.make_semantic_attributes_dataset(semantic_attributes, store='/home/runner/.config/mood/semantic_attributes', \*, n_examples=1000, batch_size=100, start_batch_idx_at=0, save_key=<function default_save_key>, verbose=0, example_generator=None, \*\*dataset_maker_kwargs)

Gather data for each semantic attribute and save it to the store.

### Parameters

- semantic_attributes: Mapping of attribute names to their descriptions
- store: Storage location (string path or MutableMapping)
- n_examples: Total number of examples to gather per attribute
- batch_size: Number of examples per batch; if None, gather all examples in one batch
- start_batch_idx_at: Starting index for batch numbering
- save_key: Function to generate save keys
- verbose: Verbosity level (0=quiet, 1=attribute progress, 2=batch progress)
- example_generator: Injectable `(attribute, n_examples, ...) -> str` callable;
  when omitted it is built lazily on `aix` (so importing `mood` stays offline)
- dataset_maker_kwargs: Additional kwargs passed to the example generator

#### WARNING
Some models aren’t very good at generating the requested number of examples.
They may generate fewer or more than requested. This is a known issue with LLM
models.

```pycon
>>> store = {}
>>> attributes = {"color": "description of color"}
>>> # A deterministic generator keeps this example offline and reproducible;
>>> # omit ``example_generator`` to use the default aix-backed LLM generator.
>>> make_semantic_attributes_dataset(
...     attributes, store, n_examples=3, batch_size=2,
...     example_generator=lambda attribute, n_examples, **_: "0 a sample segment",
... )
>>> len(store)  # Two batches saved
2
>>> store
{
    'color/color_00.txt': '0 The meeting was dull and unproductive, leaving everyone feeling dissatisfied.'
                          '2 The car was old and rusty, with a paint job that had faded over time, hinting at its once vibrant color.'
                          '5 The sunset painted the sky in brilliant shades of orange, pink, and purple, creating a breathtaking display of color that captivated all who watched.'
                          '1 The room felt gloomy, with dark furniture casting shadows, though a hint of blue could be seen in the curtains.  ',
    'color/color_01.txt': '0 The meeting was dull and uneventful, lacking any excitement or engagement.'
                          '1 The car was just an ordinary vehicle, nothing special about its appearance. '
                          "2 The flowers were vibrant, but I couldn't quite tell what colors they were. "
                          '3 The sky turned a beautiful shade of orange as the sun began to set, casting a warm glow.'
                          "4 She wore a stunning red dress that caught everyone's attention at the party."
                          '5 The ocean sparkled in shades of deep blue and turquoise, a breathtaking view that left us speechless.  '
}
```

### mood.dataset_makers.text_lines(text)

Yield the non-empty, stripped lines of `text`.

```pycon
>>> list(text_lines("a\n\n  b  \nc"))
['a', 'b', 'c']
```
