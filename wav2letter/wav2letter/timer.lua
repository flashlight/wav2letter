local Timer = torch.class('torch.FineTimer')

local ffi = require 'ffi'

ffi.cdef[[
typedef enum {
_CLOCK_REALTIME = 0,
_CLOCK_MONOTONIC = 6,
_CLOCK_MONOTONIC_RAW = 4,
_CLOCK_MONOTONIC_RAW_APPROX = 5,
_CLOCK_UPTIME_RAW = 8,
_CLOCK_UPTIME_RAW_APPROX = 9,
_CLOCK_PROCESS_CPUTIME_ID = 12,
_CLOCK_THREAD_CPUTIME_ID = 16
} clockid_t;

struct timespec
{
 long tv_sec;
 long tv_nsec;
};

 int clock_gettime(clockid_t clock_id, struct timespec *tp);
]]

function Timer:__init()
   self.__total = 0
   self:resume()
end

function Timer:reset()
   self.__total = 0
   if self.__start then
      self.__start = nil
      self:resume()
   end
end

function Timer:resume()
   if not self.__start then
      self.__start = ffi.new('struct timespec')
      assert(ffi.C.clock_gettime(0, self.__start) == 0)
   end
end

function Timer:stop()
   self.__total = self.__total + self:ticks()
   self.__start = nil
end

function Timer:time()
   return {
      real = tonumber(self.__total + self:ticks())/1000
   }
end

function Timer:ticks()
   if self.__start then
      local start = self.__start
      local now = ffi.new('struct timespec')
      ffi.C.clock_gettime(0, now)
      return (now.tv_sec - start.tv_sec) * 1000 + (now.tv_nsec - start.tv_nsec) / 1000000
   else
      return 0
   end
end

torch.Timer = torch.FineTimer
