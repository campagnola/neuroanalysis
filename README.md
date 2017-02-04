Just a Module
=============

Interactive, modular tools for analysis of neurophysiology data.

* Functions for running common analysis algorithms
* Data abstraction layer to allow adapting new data formats
* Re-usable user interface elements for implementing common analysis tasks


What makes a reusable analysis module?

- well-defined data input, result output
- functions or static methods for performing analysis
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




