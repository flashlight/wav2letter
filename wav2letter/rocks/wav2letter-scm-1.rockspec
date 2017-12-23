package = "wav2letter"
version = "scm-1"

source = {
   url = "git://github.com/facebookresearch/wav2letter.git"
}

description = {
   summary = "wav2letter toolkit",
   detailed = [[
   ]],
   homepage = "https://github.com/facebookresearch/wav2letter",
   license = "BSD+PATENTS"
}

dependencies = {
   "lua >= 5.1",
   "sndfile",
   "torchnet",
   "torchnet-sequential",
   "torchnet-optim",
   "speech",
   "gtn",
   "xlua",
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      CMAKE_PREFIX_PATH="$(LUA_BINDIR)/..",
      LUA_PATH="$(LUADIR)",
      LUA_CPATH="$(LIBDIR)",
      LIB_DIR="$(LUA_LIBDIR)",
      LUA_LIB="$(LUA_LIBDIR)/libluajit.so"
   },
   platforms = {
      macosx = {
         variables = {
            LUA_LIB="$(LUA_LIBDIR)/libluajit.dylib",
         }
      }
   }
}
