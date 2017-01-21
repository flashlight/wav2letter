local ProFi = require 'ProFi'
ProFi:start()
print("loading")
local speech = require 'speech'
print("begin")
local fun = speech.Mfcc{fs = 16000,
                        tw = 25,
                        ts = 1,
                        M  = 20,
                        N  = 13,
                        L  = 22,
                        R1 = 0,
                        R2 = 8000,
                        dev = 9}
print("continue")

for i = 1, 10 do
   print(i)
   local input = torch.rand(100000, 1)
   input = input:squeeze()
   input = fun(input, input)
end
ProFi:stop()
ProFi:writeReport( 'MyProfilingReport.txt' )
