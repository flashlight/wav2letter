package = "torchnet-sequential"
version = "scm-1"

source = {
   url = "git://github.com/torchnet/sequential.git"
}

description = {
   summary = "Sequential package for torchnet",
   detailed = [[
   Sequential package for torchnet
   ]],
   homepage = "https://github.com/torchnet/sequential",
   license = "BSD"
}

dependencies = {
   "torchnet",
   "tensorvector",
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      LUA_PATH="$(LUADIR)",
      LUA_CPATH="$(LIBDIR)"
   }
}
