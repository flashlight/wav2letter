-- (c) Ronan Collobert 2016, Facebook

local ffi = require 'ffi'
local env = {}

env.C = ffi.load(package.searchpath('libbeamer', package.cpath))

return env
