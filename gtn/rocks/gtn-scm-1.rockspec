--
-- (c) 2015 Facebook. All rights reserved.
-- Author: Ronan Collobert <locronan@fb.com>
--
--

package = "gtn"
version = "scm-1"

source = {
   url = "git://github.com/torch/gtn.git"
}

description = {
   summary = "Graph Transformer Networks for LuaJIT",
   detailed = [[
   ]],
   homepage = "https://github.com/andresy/gtn",
   license = "BSD"
}

dependencies = {
   "lua >= 5.1",
   "argcheck >= 1"
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      LUA_PATH="$(LUADIR)",
      LUA_CPATH="$(LIBDIR)"
   }
}
