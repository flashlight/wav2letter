package = "wav2letter"
version = "scm-1"

source = {
   url = "git://github.com/torch/wav2letter.git"
}

description = {
   summary = "wav2letter toolkit",
   detailed = [[
   ]],
   homepage = "https://github.com/torch/wav2letter",
   license = "FB-ONLY"
}

dependencies = {
   "lua >= 5.1",
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      CMAKE_PREFIX_PATH="$(LUA_BINDIR)/..",
      LUA_PATH="$(LUADIR)",
      LUA_CPATH="$(LIBDIR)"
   }
}
