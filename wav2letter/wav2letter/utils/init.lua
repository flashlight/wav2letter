local ffi = require 'ffi'
local argcheck = require 'argcheck'

require 'torch'

local utils = {}

-- OSS:
-- utils.C = ffi.load(package.searchpath('libwav2letter', package.cpath))
-- FBCODE:
utils.C = ffi.load('libdeeplearning_projects_wav2letter_libwav2letter.so')

local C = utils.C

ffi.cdef[[
float fccviterbi(THFloatTensor *acc, THLongTensor *macc, THFloatTensor *accp, THLongTensor *path, THFloatTensor *inp, THFloatTensor *trans, int T, int N);
double falfw(THFloatTensor *input,
              THLongTensor  *target,
              THFloatTensor *trans,
              THDoubleTensor *acc,
              THLongTensor *macc,
              int64_t T,
              int64_t N,
              int64_t TN);
double fccfw(THFloatTensor *input,
              THFloatTensor *trans,
              THLongTensor *macc,
              THDoubleTensor *acc,
              int64_t T,
              int64_t N);
void falbw(THFloatTensor *input,
              THLongTensor  *target,
              THFloatTensor *trans,
              THFloatTensor *gem,
              THFloatTensor *gtrans,
              THDoubleTensor *acc,
              THDoubleTensor *gacc,
              float scale,
              int64_t T,
              int64_t N,
              int64_t TN);
void fccbw(THFloatTensor *input,
              THFloatTensor *trans,
              THFloatTensor *gem,
              THFloatTensor *gtrans,
              THLongTensor *macc,
              THDoubleTensor *acc,
              THDoubleTensor *gacc,
              float scale,
              int64_t T,
              int64_t N);
void asgbatchfw(THFloatTensor **input,
              THLongTensor  **target,
              THFloatTensor *trans,
              THDoubleTensor *falacc,
              THLongTensor *falmacc,
              THDoubleTensor *falgacc,
              THDoubleTensor *fccacc,
              THLongTensor *fccmacc,
              THDoubleTensor *fccgacc,
              THFloatTensor  *falscale,
              THFloatTensor  *fccscale,
              THLongTensor  *T,
              int64_t N,
              THLongTensor  *TN,
              THFloatTensor  *loss,
              int64_t B);
void asgbatchbw(THFloatTensor **input,
              THLongTensor  **target,
              THFloatTensor *trans,
              THFloatTensor *falgem,
              THFloatTensor *fccgem,
              THFloatTensor *falgtrans,
              THFloatTensor *fccgtrans,
              THDoubleTensor *falacc,
              THLongTensor *falmacc,
              THDoubleTensor *falgacc,
              THDoubleTensor *fccacc,
              THLongTensor *fccmacc,
              THDoubleTensor *fccgacc,
              THFloatTensor  *falscale,
              THFloatTensor  *fccscale,
              THLongTensor  *T,
              int64_t N,
              THLongTensor  *TN,
              int64_t B);
]]

for _, Real in ipairs{'Byte', 'Short', 'Int', 'Long'} do
   local cdef = [[
void THRealTensor_reduceMostFrequentIndex(THRealTensor *dst, THRealTensor *src, int dimension, int N);
long THRealTensor_editdistance(THRealTensor *s, THRealTensor *t);
void THRealTensor_uniq(THRealTensor *dst, THRealTensor *src);
void THRealTensor_replabel(THRealTensor *dst, THRealTensor *src, int replabel, int nclass);
void THRealTensor_invreplabel(THRealTensor *dst, THRealTensor *src, int replabel, int nclass);
]]
   cdef = cdef:gsub('Real', Real)
   ffi.cdef(cdef)

   local RealTensor = torch[Real .. "Tensor"]
   local RealTensorReal = "torch." .. Real .. "Tensor"

   utils.mostfrequentindex = argcheck{
      {name="dst", type=RealTensorReal},
      {name="src", type=RealTensorReal},
      {name="dim", type="number"},
      {name="N", type="number"},
      overload = utils.mostfrequentindex,
      call =
         function(dst, src, dim, N)
            C["TH" .. Real .. "Tensor_reduceMostFrequentIndex"](dst:cdata(), src:cdata(), dim-1, N)
         end
   }

   utils.editdistance = argcheck{
      {name="s", type=RealTensorReal},
      {name="t", type=RealTensorReal},
      overload = utils.mostfrequentindex,
      call =
         function(s, t)
            return tonumber(
               C["TH" .. Real .. "Tensor_editdistance"](s:cdata(), t:cdata()))
         end
   }

   utils.uniq = argcheck{
      {name="dst", type=RealTensorReal, opt=true},
      {name="src", type=RealTensorReal},
      overload = utils.uniq,
      call =
         function(dst, src)
            dst = dst or RealTensor()
            C["TH" .. Real .. "Tensor_uniq"](dst:cdata(), src:cdata())
            return dst
         end
   }

   utils.replabel = argcheck{
      {name="src", type=RealTensorReal},
      {name="rep", type='number'},
      {name="nclass", type='number'},
      {name="dst", type=RealTensorReal, opt=true},
      overload = utils.replabel,
      call =
         function(src, rep, nclass, dst)
            dst = dst or RealTensor()
            C["TH" .. Real .. "Tensor_replabel"](dst:cdata(), src:cdata(), rep, nclass)
            return dst
         end
   }

   utils.invreplabel = argcheck{
      {name="src", type=RealTensorReal},
      {name="rep", type='number'},
      {name="nclass", type='number'},
      {name="dst", type=RealTensorReal, opt=true},
      overload = utils.invreplabel,
      call =
         function(src, rep, nclass, dst)
            dst = dst or RealTensor()
            C["TH" .. Real .. "Tensor_invreplabel"](dst:cdata(), src:cdata(), rep, nclass)
            return dst
         end
   }
end

return utils
