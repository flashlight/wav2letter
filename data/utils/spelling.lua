-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.

-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

-- handles repeated letters
return function(spelling, maxr)
   if spelling:match('%d') then
      error('invalid spelling <' .. spelling .. '> (no number allowed when handling repeated letters)')
   end
   local prevletter = spelling:sub(1, 1)
   local newspelling = prevletter
   local r = 0
   local invalid = false
   for i=2,#spelling do
      local letter = spelling:sub(i, i)
      if letter == prevletter then
         r = r + 1
         if r == maxr then
            newspelling = newspelling .. r
            prevletter = nil
            r = 0
            invalid = true
         elseif i == #spelling then
            newspelling = newspelling .. r
         end
      else
         if r >= 1 then
            newspelling = newspelling .. r
         end
         r = 0
         prevletter = letter
         newspelling = newspelling .. letter
      end
   end
   if invalid then
      print('! invalid spelling <' .. spelling .. '> converted to <' .. newspelling .. '>')
   end
   return newspelling
end
