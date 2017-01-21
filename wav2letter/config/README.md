config must contain:

  - config.traindataset: a function which returns a dataset
  - config.validdatasets: a hash table of valid set names associated to a function returning a dataset
  - config.testdatasets: a hash table of valid set names associated to a function returning a dataset
  - config.transforms: a table which _may_ contain:
    - transforms.input: transformation function applied to input
    - transforms.target: transformation function applied to target
    - transforms.remap: transformation function applied to output and target for eval purposes
  - config.tostring: a function which prints eval output / target
  - config.specs: a table containing basic specs:
    - specs.nchannel: number of channels in audio
    - specs.samplerate: audio sample rate
    - specs.nclass: number of classes
