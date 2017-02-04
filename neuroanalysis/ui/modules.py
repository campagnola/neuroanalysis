"""
Modular analysis classes:

- data input, result output
- static method(s) for performing analysis
    - standard options for requesting more comprehansive output / intermediate values
        Module.process(return=['event_arrays', 'fit_errors'])
- optional user interface
    - control widgets
        -> based on parametertree
    - plot / graphicsview adapters
        -> easy disconnect / reconnect from plots
    - save/restore state
    - display previously-generated results (to avoid recomputation)
    - display results read-only
    - create flowchart node
    - how to provide multiple visual outputs with a single control input?
        -> create UI with shared parameters.
"""




