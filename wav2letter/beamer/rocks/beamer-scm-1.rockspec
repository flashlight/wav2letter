package = "beamer"
version = "scm-1"

source = {
   url = "git://github.com/torch/beamer.git"
}

description = {
   summary = "Large Vocabulary Beam Search",
   detailed = [[
   Large Vocabulary Beam Search
   ]],
   homepage = "https://github.com/torch/beamer",
   license = "BSD"
}

dependencies = {
   "lua >= 5.1",
   "torch >= 7.0",
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      CMAKE_PREFIX_PATH="$(LUA_BINDIR)/..",
      CMAKE_INSTALL_PREFIX="$(PREFIX)"
   }
}
