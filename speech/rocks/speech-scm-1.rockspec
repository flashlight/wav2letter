package = "speech"
version = "scm-1"

source = {
   url = "git://github.com/torch/speech.git"
}

description = {
   summary = "speech processing",
   detailed = [[
   speech processing
   ]],
   homepage = "https://github.com/torch/speech",
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
