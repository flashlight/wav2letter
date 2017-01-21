-- (c) Ronan Collobert 2016, Facebook

local ffi = require 'ffi'
local env = {}
-- OSS: env.C = ffi.load(package.searchpath('libbeamer', package.cpath))
-- SUX:
env.C = ffi.load('deeplearning_projects_wav2letter_libbeamer')
return env
