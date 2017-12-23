package = "torchnet-optim"
version = "scm-1"

source = {
   url = "git://github.com/torchnet/optim.git"
}

description = {
   summary = "Optim package for torchnet",
   detailed = [[
   Optim package for torchnet
   ]],
   homepage = "https://github.com/torchnet/optim",
   license = "BSD"
}

dependencies = {
   "torchnet",
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      LUA_PATH="$(LUADIR)",
      LUA_CPATH="$(LIBDIR)"
   }
}
