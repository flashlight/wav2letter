package = "tensorvector"
version = "scm-1"

source = {
   url = "git://github.com/torch/tensorvector.git"
}

description = {
   summary = "Tensorvector package for torch",
   detailed = [[
   Tensorvector package for torchnet
   ]],
   homepage = "https://github.com/torch/tensorvector",
   license = "BSD"
}

dependencies = {
   "torch",
   "tds",
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      LUA_PATH="$(LUADIR)",
      LUA_CPATH="$(LIBDIR)"
   }
}
